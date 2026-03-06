import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Credit Risk Engine", layout="wide")
st.title("Financial Risk Scoring Dashboard")

st.sidebar.header("Applicant Data")

# Input fields matching strict Pydantic schema bounds
limit_bal = st.sidebar.number_input(
    "Credit Limit (LIMIT_BAL)", min_value=1.0, value=50000.0
)
sex = st.sidebar.selectbox("Sex (1=Male, 2=Female)", [1, 2])
education = st.sidebar.selectbox(
    "Education (1=Grad, 2=Uni, 3=HS, 4=Other)", [1, 2, 3, 4]
)
marriage = st.sidebar.selectbox("Marriage (1=Married, 2=Single, 3=Other)", [1, 2, 3])
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)

st.sidebar.subheader("Repayment Status (-1=Pay duly, 1+=Months Delay)")
pay_0 = st.sidebar.slider("PAY_0 (September)", -2, 9, 0)
pay_2 = st.sidebar.slider("PAY_2 (August)", -2, 9, 0)
pay_3 = st.sidebar.slider("PAY_3 (July)", -2, 9, 0)
pay_4 = st.sidebar.slider("PAY_4 (June)", -2, 9, 0)
pay_5 = st.sidebar.slider("PAY_5 (May)", -2, 9, 0)
pay_6 = st.sidebar.slider("PAY_6 (April)", -2, 9, 0)

st.sidebar.subheader("Bill Amounts & Payments")
bill_amt1 = st.sidebar.number_input("BILL_AMT1", value=0.0)
pay_amt1 = st.sidebar.number_input("PAY_AMT1", min_value=0.0, value=1000.0)

if st.sidebar.button("Run Risk Assessment"):
    # Construct strictly typed payload
    payload = {
        "LIMIT_BAL": 500000.0,
        "SEX": 2,
        "EDUCATION": 1,
        "MARRIAGE": 2,
        "AGE": 30,
        "PAY_0": -1,
        "PAY_2": -1,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -1,
        "PAY_6": -1,
        "BILL_AMT1": 5000.0,
        "BILL_AMT2": 5000.0,
        "BILL_AMT3": 5000.0,
        "BILL_AMT4": 5000.0,
        "BILL_AMT5": 5000.0,
        "BILL_AMT6": 5000.0,
        "PAY_AMT1": 5000.0,
        "PAY_AMT2": 5000.0,
        "PAY_AMT3": 5000.0,
        "PAY_AMT4": 5000.0,
        "PAY_AMT5": 5000.0,
        "PAY_AMT6": 5000.0,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        st.subheader("Decision Output")

        col1, col2, col3 = st.columns(3)
        col1.metric("Decision", result["decision"])
        col2.metric("Probability of Default", f"{result['probability']:.4f}")
        col3.metric("Risk Tier", result["risk_tier"])

        st.text(f"Applied Economic Threshold: {result['threshold_applied']}")

    except requests.exceptions.ConnectionError:
        st.error(
            "API Connection Failed. Ensure the Docker container is running on port 8000."
        )
    except requests.exceptions.HTTPError:
        st.error(f"API Schema Validation Error: {response.text}")
