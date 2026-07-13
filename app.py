import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection System", page_icon="💳")

st.title("💳 Fraud Detection System")
st.write("Upload a CSV file containing digital payment transactions.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df)

    if "Amount" not in df.columns:
        st.error("CSV file must contain a column named 'Amount'")
    else:

        model = IsolationForest(contamination=0.2, random_state=42)

        df["Prediction"] = model.fit_predict(df[["Amount"]])

        df["Result"] = df["Prediction"].map({1: "Normal", -1: "Fraud"})

        st.subheader("Fraud Detection Result")
        st.dataframe(df)

        fraud_count = (df["Result"] == "Fraud").sum()
        normal_count = (df["Result"] == "Normal").sum()

        st.success(f"✅ Normal Transactions: {normal_count}")
        st.error(f"🚨 Fraud Transactions: {fraud_count}")

        result_count = df["Result"].value_counts()

        fig, ax = plt.subplots()
        result_count.plot(kind="bar", ax=ax)
        ax.set_xlabel("Transaction Type")
        ax.set_ylabel("Count")
        ax.set_title("Fraud Detection Results")

        st.pyplot(fig)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Result CSV",
            data=csv,
            file_name="fraud_detection_result.csv",
            mime="text/csv",
        )
