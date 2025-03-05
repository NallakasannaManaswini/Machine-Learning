import streamlit as st
import pickle
import numpy as np

# Load models
le_model = pickle.load(open('le.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
xgb_model = pickle.load(open('xgb.pkl', 'rb'))

# Custom CSS for compact design
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #FFD89C, #FF9A8B);
            color: #333333;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #FF6F61;
        }
        .stButton>button {
            background: linear-gradient(45deg, #FF6F61, #FF914D);
            color: #1E3A8A !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #FF914D, #FF6F61);
        }
        .custom-header {
            font-size: 26px;
            font-weight: bold;
            color: #D72638;
            text-align: center;
            margin-bottom: 10px;
        }
        .prediction-box {
            background: #28A745;
            color: white;
            font-size: 20px;
            font-weight: bold;
            padding: 10px;
            text-align: center;
            border-radius: 8px;
            margin-top: 15px;
            box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("<div class='custom-header'>🏡 House Price Prediction</div>", unsafe_allow_html=True)

# More compact column layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    garage = st.text_input("Garage *", placeholder="Enter number of garages")
    land_area = st.text_input("Land Area (sq ft) *", placeholder="Enter land area")
    floor_area = st.text_input("Floor Area (sq ft) *", placeholder="Enter floor area")
    cbd_dist = st.text_input("Distance to CBD (m) *", placeholder="Enter distance")

with col2:
    nearest_stn_dist = st.text_input("Distance to Nearest Station (m) *", placeholder="Enter distance")
    postcode = st.text_input("Postcode *", placeholder="Enter postcode")
    nearest_sch = st.text_input("Nearest School", placeholder="Enter school name")
    nearest_sch_dist = st.text_input("Distance to Nearest School (m) *", placeholder="Enter distance")

with col3:
    month_sold = st.text_input("Month Sold *", placeholder="Enter month (1-12)")
    year_sold = st.text_input("Year Sold *", placeholder="Enter year")
    age = st.text_input("Age of Property (years) *", placeholder="Enter age")
    no_of_rooms = st.text_input("Number of Rooms *", placeholder="Enter rooms")

# Convert inputs
try:
    garage = int(garage) if garage else None
    land_area = float(land_area) if land_area else None
    floor_area = float(floor_area) if floor_area else None
    cbd_dist = float(cbd_dist) if cbd_dist else None
    nearest_stn_dist = float(nearest_stn_dist) if nearest_stn_dist else None
    postcode = int(postcode) if postcode else None
    nearest_sch_dist = float(nearest_sch_dist) if nearest_sch_dist else None
    month_sold = int(month_sold) if month_sold else None
    year_sold = int(year_sold) if year_sold else None
    age = int(age) if age else None
    no_of_rooms = int(no_of_rooms) if no_of_rooms else None
except ValueError:
    st.error("❌ Please enter valid numeric values.")
    st.stop()

# Validation
if None in [garage, land_area, floor_area, cbd_dist, nearest_stn_dist, postcode, nearest_sch_dist, month_sold, year_sold, age, no_of_rooms]:
    st.warning("⚠️ Fill in all required fields.")
else:
    if nearest_sch.strip() == "":
        nearest_sch = "Unknown School"
    if nearest_sch not in le_model.classes_:
        le_model.classes_ = np.append(le_model.classes_, nearest_sch)
    nearest_sch_encoded = le_model.transform([nearest_sch])[0]

    features = [
        garage, land_area, floor_area, cbd_dist, nearest_stn_dist, postcode,
        nearest_sch_encoded, nearest_sch_dist, month_sold, year_sold, age, no_of_rooms
    ]
    ind = [1, 2, 3, 7, 4, 6]

    features = np.array(features).reshape(1, -1)
    try:
        features[:, ind] = scaler.transform(features[:, ind])
        if st.button("🔮 Predict Price"):
            prediction = xgb_model.predict(features)[0]
            st.markdown(f"<div class='prediction-box'>💰 Predicted House Price: ₹{prediction:,.2f}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
