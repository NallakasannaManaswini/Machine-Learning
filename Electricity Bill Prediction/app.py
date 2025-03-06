import streamlit as st
import pickle
import numpy as np

st.set_page_config(layout="wide")

# Load the pickled models and encoders
le_city = pickle.load(open('le_city.pkl', 'rb'))
le_company = pickle.load(open('le_company.pkl', 'rb'))
standard_scaler = pickle.load(open('scaler.pkl', 'rb'))
lin_model = pickle.load(open('lin.pkl', 'rb'))

# Custom CSS to change the slider color to blue and add background gradient
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #E0F7FA, #80DEEA);
        color: #333333;
    }
    .stSlider > div > div > div > div {
        background-color: blue;
        
    }
    .stSlider > div > div > div > div[data-baseweb="slider-thumb"] {
        background-color: blue;
    }
    .stButton > button {
        background-color:#0D47A1; 
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Centered title with emoji
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center;">
    <h1 style="color: #0D47A1; font-size: 48px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
        ⚡ Electricity Bill Prediction ⚡
    </h1>
</div>
    """,
    unsafe_allow_html=True,
)

# Creating three columns
col1, col2, col3 = st.columns(3)

# Input fields with h2 headers
with col1:
    st.markdown("<h4>Fan</h4>", unsafe_allow_html=True)
    fan = st.slider("", min_value=0, max_value=50, value=10)
    st.markdown("<h4>Television</h4>", unsafe_allow_html=True)
    television = st.slider("", min_value=0, max_value=50, value=5)
    st.markdown("<h4>Month</h4>", unsafe_allow_html=True)
    month = st.slider("", min_value=1, max_value=12, value=6)

with col2:
    st.markdown("<h4>Refrigerator</h4>", unsafe_allow_html=True)
    refrigerator = st.slider("", min_value=0, max_value=50, value=2)
    st.markdown("<h4>Monitor</h4>", unsafe_allow_html=True)
    monitor = st.slider("", min_value=0, max_value=20, value=3)
    st.markdown("<h4>City</h4>", unsafe_allow_html=True)
    city = st.selectbox("", options=le_city.classes_)

with col3:
    st.markdown("<h4>Air Conditioner</h4>", unsafe_allow_html=True)
    air_conditioner = st.slider("", min_value=0, max_value=10, value=1)
    st.markdown("<h4>Company</h4>", unsafe_allow_html=True)
    company = st.selectbox("", options=le_company.classes_)
    st.markdown("<h4>Monthly Hours</h4>", unsafe_allow_html=True)
    monthly_hours = st.slider("", min_value=100, max_value=1000, value=500)

# Transform categorical inputs
city_encoded = int(le_city.transform([city]))
company_encoded = int(le_company.transform([company]))

# Scale numerical inputs
scaled_values = standard_scaler.transform([[television, monthly_hours]])[0]
television_scaled, monthly_hours_scaled = scaled_values

# Prepare data for prediction
data = np.array([fan, refrigerator, air_conditioner, television_scaled, monitor, month, city_encoded, company_encoded, monthly_hours_scaled]).reshape(1, -1)

# Prediction button
if st.button("Predict Electricity Usage"):
    prediction = lin_model.predict(data)[0]
    st.markdown(
        f"""
        <div style="background-color: #4CAF50; color: white; padding: 15px; text-align: center; border-radius: 10px;">
            <h3 style="margin: 0; color: white;">Predicted Electricity Bill in Rs: ₹ {prediction:.2f} </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
