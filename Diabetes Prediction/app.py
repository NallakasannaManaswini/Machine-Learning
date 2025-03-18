import streamlit as st
import streamlit as st
import pickle
import numpy as np

# Load models
oe_model = pickle.load(open("oe.pkl", "rb"))
logistic_model = pickle.load(open("logistic.pkl", "rb"))

# Set full width for the page
st.set_page_config(layout="wide")

# Custom UI Styling
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            color: #6A0572;
            font-weight: bold;
            padding-bottom: 20px;
        }

        .stTextInput label, .stSelectbox label {
            font-weight: bold !important;
            font-size: 22px !important;
        }

        .stNumberInput input, .stSelectbox select {
            font-size: 20px !important;
            padding: 12px !important;
            border-radius: 8px;
        }

        .stButton>button {
            background-color: #6A0572;
            color: white;
            border-radius: 12px;
            padding: 14px 35px;
            font-size: 22px;
            font-weight: bold;
            margin-top: 25px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #A40E4C;
        }

        .prediction {
            font-size: 26px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            margin-top: 25px;
            width: 60%;
            margin-left: auto;
            margin-right: auto;
        }
        .diabetic { background-color: #FFCDD2; color: red; }
        .non-diabetic { background-color: #C8E6C9; color: green; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="title">🩺 Smart Diabetes Prediction System 🔬</div>', unsafe_allow_html=True)

# Function to get user input
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    age = st.number_input("**Age (Years)**", min_value=0, max_value=120, step=1)
    bmi = st.number_input("**BMI**", min_value=0, max_value=300, step=1)
    fpg = st.number_input("**Fasting Plasma Glucose (FPG)**", min_value=0.0, max_value=50.0, step=0.1)
    hdl = st.number_input("**HDL (Good Cholesterol)**", min_value=0.0, max_value=50.0, step=0.1)
    ccr = st.number_input("**Creatinine Clearance Rate (CCR)**", min_value=0.0, max_value=50.0, step=0.1)

with col2:
    gender = st.selectbox("**Gender**", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    systolic_bp = st.number_input("**Systolic BP (SBP) mmHg**", min_value=50, max_value=200, step=1)
    chol = st.number_input("**Total Cholesterol (mg/dL)**", min_value=0.0, max_value=50.0, step=0.1)
    ldl = st.number_input("**LDL (Bad Cholesterol)**", min_value=0.0, max_value=50.0, step=0.1)
    ffpg = st.number_input("**Follow-up FPG**", min_value=0.0, max_value=50.0, step=0.1)

with col3:
    diastolic_bp = st.number_input("**Diastolic BP (DBP) mmHg**", min_value=0.0, max_value=100.0, step=0.5)
    tri = st.number_input("**Triglycerides (mg/dL)**", min_value=0.0, max_value=100.0, step=0.1)
    alt = st.number_input("**ALT (Alanine Aminotransferase)**", min_value=0.0, max_value=50.0, step=0.1)
    bun = st.number_input("**BUN (Blood Urea Nitrogen)**", min_value=0.0, max_value=50.0, step=0.1)

# Categorical Inputs
col_smoking, col_drinking = st.columns(2)

with col_smoking:
    smoking = st.selectbox("**Smoking Status 🚬**", ["Never", "Past", "Occational", "Daily"], key="smoking")

with col_drinking:
    drinking = st.selectbox("**Drinking Status 🍷**", ["Never", "Past", "Occational", "Daily"], key="drinking")

# Convert categorical features
encoded_smoking = oe_model.transform([[smoking]]).flatten().tolist()
encoded_drinking = oe_model.transform([[drinking]]).flatten().tolist()

# Ensure all inputs are valid before predicting
if None not in [age, bmi, fpg, hdl, ccr, systolic_bp, chol, ldl, ffpg, diastolic_bp, tri, alt, bun]:
    data = [age, gender, bmi, systolic_bp, diastolic_bp, fpg, chol, tri, hdl, ldl, alt, bun, ccr, ffpg] + encoded_smoking + encoded_drinking
    X_input = np.array(data, dtype=np.float32).reshape(1, -1)

    # Centering the button
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    if st.button("🔍 Predict Diabetes"):
        prediction = logistic_model.predict(X_input)[0]
        if prediction == 1:
            result_text = "Diabetic 🩸"
            color_class = "diabetic"
        else:
            result_text = "Non-Diabetic ✅"
            color_class = "non-diabetic"

        st.markdown(f'<div class="prediction {color_class}">You are {result_text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
