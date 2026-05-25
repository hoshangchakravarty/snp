import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import time

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(layout="wide")

st.title("🚨 AlphaFinder: BiLSTM + Dropout Regime-Aware Sentiment Forecasting for S&P 500")

st.markdown("""
This app helps identify when news sentiment actually matters for predicting stock movement.
By focusing on specific event types like layoffs or earnings and company media exposure levels
(low, mid, high), we train a BiLSTM model to predict whether a stock will be among the top
performers over the next 3 days.

Instead of guessing blindly, this model spots high-signal regimes where sentiment has true
forecasting power.

📅 The dataset covers daily stock news and prices for S&P 500 companies from **October 2020 to July 2022**.
""")

# -----------------------------
# Company examples for summary
# -----------------------------
coverage_companies = {
    "Low Coverage": ["ConAgra Brands", "AutoNation", "Clorox"],
    "Mid Coverage": ["Adobe", "Ford", "Cisco"],
    "High Coverage": ["Apple", "Tesla", "Amazon"]
}

# -----------------------------
# Load and prepare data
# -----------------------------
@st.cache_data
def load_data():
    # Reads compressed CSV directly
    df = pd.read_csv("data.csv.gz", parse_dates=["Date"])

    # Sentiment feature
    df["Net Sentiment"] = (
        df["News - Positive Sentiment"] - df["News - Negative Sentiment"]
    )

    # Sort by stock and date
    df = df.sort_values(by=["Symbol", "Date"])

    # Future 3-day return
    df["3D Return"] = df.groupby("Symbol")["Close"].shift(-3) / df["Close"] - 1

    # Previous 1-day return
    df["Lag1 Return"] = df.groupby("Symbol")["Close"].pct_change()

    # Rolling sentiment
    df["rolling_sentiment_3d"] = (
        df.groupby("Symbol")["Net Sentiment"]
        .transform(lambda x: x.rolling(3).mean())
    )

    # Daily news count proxy
    df["news_volume"] = df.groupby(["Symbol", "Date"])["Net Sentiment"].transform("count")

    # Coverage classification using average News Volume
    coverage_map_df = df.groupby("Symbol")["News - Volume"].mean()

    try:
        coverage_class_df = pd.qcut(
            coverage_map_df,
            q=3,
            labels=["Low Coverage", "Mid Coverage", "High Coverage"],
            duplicates="raise"
        )
    except ValueError:
        coverage_class_df = pd.cut(
            coverage_map_df,
            bins=[
                coverage_map_df.min() - 1e-6,
                coverage_map_df.median(),
                coverage_map_df.max()
            ],
            labels=["Low Coverage", "High Coverage"]
        )

    df["Coverage Class"] = df["Symbol"].map(coverage_class_df)
    df = df[df["Coverage Class"].notna()]

    labels_present = sorted(df["Coverage Class"].unique())

    return df, labels_present


def create_sequences(X, y, window=3):
    Xs, ys = [], []

    for i in range(len(X) - window + 1):
        Xs.append(X[i:i + window])
        ys.append(y[i + window - 1])

    return np.array(Xs), np.array(ys)


df, valid_coverage_classes = load_data()

# -----------------------------
# Event columns
# -----------------------------
event_cols = [
    "News - Corporate Earnings",
    "News - Mergers & Acquisitions",
    "News - Layoffs",
    "News - Product Recalls",
    "News - Adverse Events",
    "News - Personnel Changes",
    "News - Stocks"
]

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📊 Regime Selection")

    selected_event = st.selectbox("Event Type", event_cols)
    selected_coverage = st.selectbox("Coverage Class", valid_coverage_classes)

    st.markdown("---")
    st.subheader("⚙️ Model Settings")

    num_epochs = st.slider(
        "Epochs",
        min_value=10,
        max_value=100,
        value=30,
        step=5
    )

    batch_size = st.slider(
        "Batch Size",
        min_value=16,
        max_value=128,
        value=32,
        step=16
    )

    train_button = st.button("🚀 Train LSTM")

# -----------------------------
# Main training logic
# -----------------------------
if train_button:
    clean_event_name = selected_event.replace("News - ", "")

    st.subheader(
        f"Training LSTM on regime: {clean_event_name} | {selected_coverage}"
    )

    df_filtered = df[
        (df[selected_event] > 0) &
        (df["Coverage Class"] == selected_coverage)
    ].copy()

    df_filtered = df_filtered.dropna(
        subset=[
            "Net Sentiment",
            "3D Return",
            "Lag1 Return",
            "rolling_sentiment_3d",
            "news_volume"
        ]
    )

    # Safety check: not enough rows
    if len(df_filtered) < 30:
        st.error(
            f"Not enough data for this regime. Only {len(df_filtered)} usable rows found. "
            "Try another event type or coverage class."
        )
        st.stop()

    # Create label: top 34% future 3-day returns
    threshold = df_filtered["3D Return"].quantile(0.66)
    df_filtered["label"] = (df_filtered["3D Return"] >= threshold).astype(int)

    df_pos = df_filtered[df_filtered["label"] == 1]
    df_neg = df_filtered[df_filtered["label"] == 0]

    # Safety check: one class missing
    if len(df_pos) == 0 or len(df_neg) == 0:
        st.error(
            "This regime does not have both positive and negative classes after labeling. "
            "Try another regime."
        )
        st.stop()

    # Balance dataset by upsampling minority/positive class to negative class size
    df_pos_upsampled = resample(
        df_pos,
        replace=True,
        n_samples=len(df_neg),
        random_state=42
    )

    df_balanced = pd.concat([df_neg, df_pos_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    features = [
        "Net Sentiment",
        "Lag1 Return",
        "rolling_sentiment_3d",
        "news_volume"
    ]

    X = df_balanced[features].values
    y = df_balanced["label"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = create_sequences(X_scaled, y, window=3)

    # Safety check: sequence count
    if len(X_seq) < 20:
        st.error(
            f"Not enough sequence data after applying the 3-day window. "
            f"Only {len(X_seq)} sequences available."
        )
        st.stop()

    # Safety check: stratification possible
    unique_classes, class_counts = np.unique(y_seq, return_counts=True)

    if len(unique_classes) < 2 or np.min(class_counts) < 2:
        st.error(
            "Not enough class balance after sequence creation for stratified train-test split. "
            "Try another regime."
        )
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq,
        y_seq,
        test_size=0.2,
        random_state=42,
        stratify=y_seq
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = Sequential()

    model.add(
        Bidirectional(
            LSTM(64, activation="tanh"),
            input_shape=(X_seq.shape[1], X_seq.shape[2])
        )
    )

    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # -----------------------------
    # Training
    # -----------------------------
    with st.spinner("Training model..."):
        start = time.time()

        history = model.fit(
            X_train,
            y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )

        end = time.time()

    st.success(f"Training completed in {end - start:.2f} seconds!")

    # -----------------------------
    # Prediction and metrics
    # -----------------------------
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = np.mean(y_pred == y_test)

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    company_list = coverage_companies.get(selected_coverage, [])
    company_str = ", ".join(company_list) if company_list else "selected companies"

    # -----------------------------
    # Summary cards
    # -----------------------------
    if acc > 0.5:
        st.success(
            f"✅ Summary: {selected_coverage} stocks like {company_str} on "
            f"{clean_event_name} news show an edge with accuracy {acc:.2f}."
        )

        st.markdown(f"""
        <div style="
            background-color:#dcfce7;
            color:#14532d;
            padding:1rem;
            border-left:6px solid #16a34a;
            border-radius:0.5rem;
            font-size:1rem;
            line-height:1.6;
            margin-top:0.5rem;
            margin-bottom:1rem;
        ">
            <b style="color:#052e16;">📌 What this means:</b><br>
            Stocks like <b>{company_str}</b> in the <b>{selected_coverage}</b> group,
            when affected by <b>{clean_event_name}</b> news, tend to perform better
            over the next 3 days. The model identifies such patterns with
            <b>{acc:.0%}</b> accuracy, offering a statistical edge for strategy filtering.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info(
            f"⚠️ Summary: {selected_coverage} stocks like {company_str} show limited "
            f"predictive power on {clean_event_name} news with accuracy {acc:.2f}."
        )

        st.markdown(f"""
        <div style="
            background-color:#fff8db;
            color:#1f2937;
            padding:1rem;
            border-left:6px solid #f59e0b;
            border-radius:0.5rem;
            font-size:1rem;
            line-height:1.6;
            margin-top:0.5rem;
            margin-bottom:1rem;
        ">
            <b style="color:#111827;">📌 What this means:</b><br>
            In this regime, sentiment alone is not a reliable predictor.
            For stocks like <b>{company_str}</b>, additional signals may be required
            to improve confidence.
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    st.subheader("📌 Confusion Matrix")

    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # -----------------------------
    # Training loss
    # -----------------------------
    st.subheader("📉 Training Loss")

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(history.history["loss"], label="Train Loss")
    ax_loss.plot(history.history["val_loss"], label="Validation Loss")
    ax_loss.set_title("Loss over Epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Binary Crossentropy Loss")
    ax_loss.legend()
    st.pyplot(fig_loss)

    # -----------------------------
    # Accuracy
    # -----------------------------
    st.subheader("✅ Test Accuracy")
    st.write(f"**{acc:.4f}**")

    # -----------------------------
    # Classification report
    # -----------------------------
    st.subheader("📋 Classification Report")

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.4f}"))

    # -----------------------------
    # Feature influence proxy
    # -----------------------------
    st.subheader("📊 Feature Influence Proxy")

    st.caption(
        "This is a simple mean absolute value proxy, not true feature importance. "
        "For a neural network, proper feature importance would require permutation importance, "
        "SHAP, or ablation testing."
    )

    importances = np.mean(np.abs(df_balanced[features]), axis=0)

    fig_feat, ax_feat = plt.subplots()
    ax_feat.barh(features, importances)
    ax_feat.set_title("Feature Influence Proxy")
    ax_feat.set_xlabel("Mean Absolute Value")
    st.pyplot(fig_feat)

else:
    st.info("Select a regime and press the button to train an LSTM model.")