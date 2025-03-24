import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load models
le_model = pickle.load(open('le.pkl', 'rb'))
rf_model = pickle.load(open('rf.pkl', 'rb'))

# Feature names
feature_names = ['Hours', 'Mins', 'Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
       'Educational_level', 'Vehicle_driver_relation', 'Driving_experience',
       'Type_of_vehicle', 'Owner_of_vehicle', 'Area_accident_occured',
       'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction',
       'Road_surface_type', 'Road_surface_conditions', 'Light_conditions',
       'Weather_conditions', 'Type_of_collision',
       'Number_of_vehicles_involved', 'Number_of_casualties',
       'Vehicle_movement', 'Cause_of_accident']

# Indexes that need encoding
inx = [2,4,5,6,8,9,10,11,12,13,14,15,16,17,18,21,22]

# Mapping dictionaries
ages = {'Under 18': 0, '18-30': 1, '31-50': 2, 'Over 51': 3, 'Unknown': -1}
exp = {'No Licence': 0, 'Below 1yr': 1, '1-2yr': 2, '2-5yr': 3, '5-10yr': 4, 'Above 10yr': 5, 'unknown': -1}

# Streamlit UI
st.set_page_config(layout="wide")

st.markdown(
    """
    <h1 style="text-align: center; color: #9ACD32;">
        🚦 Accident Severity Prediction 🚑
    </h1>
    """,
    unsafe_allow_html=True
)


st.markdown(
"""
<style>
body {
    background-color: #CCE4E2; /* Light Pink Background */
    color: black; /* Black text for contrast */
}
.stApp {
    background-color: #CCE4E2;
}
div.stButton > button {
    background-color: #4CAF50; /* Green */
    color: white;
    padding: 10px 30px;
    font-size: 20px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #45a049;
}
</style>
""",
unsafe_allow_html=True
)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

data = []
with col1:
    data.append(st.number_input("Hours ⏳", min_value=0, max_value=23, value=12))
    data.append(st.number_input("Minutes ⏰", min_value=0, max_value=59, value=20))
    data.append(st.selectbox("Day of Week 📅", ['Sunday','Monday','Tuesday' ,'Wednesday','Thursday', 'Friday', 'Saturday' ], index=0))
    data.append(st.selectbox("Age Band of Driver 👨‍🦳", list(ages.keys()), index=0))
    data.append(st.selectbox("Sex of Driver 🚻", ['Male', 'Female'], index=0))

    
with col2:
    data.append(st.selectbox("Educational Level 🎓", 
        ['Above high school','High school', 'Junior high school', 'Elementary school', 'Writing & reading', 'Illiterate'], index=0))

    data.append(st.selectbox("Vehicle Driver Relation 🚗", 
        ['Owner','Employee', 'Other'], index=1))

    data.append(st.selectbox("Driving Experience 🏎️", 
        list(exp.keys()), index=2))

    data.append(st.selectbox("Type of Vehicle 🚙", 
        ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)', 'Public (13?45 seats)', 'Lorry (11?40Q)', 'Long lorry', 
        'Public (12 seats)', 'Taxi', 'Pick up upto 10Q', 'Stationwagen', 'Ridden horse', 'Bajaj', 'Turbo', 'Motorcycle', 
        'Special vehicle', 'Bicycle', 'Other'], index=0))

    data.append(st.selectbox("Owner of Vehicle 🏠", 
        ['Owner', 'Governmental', 'Organization', 'Other'], index=0))

    data.append(st.selectbox("Area Accident Occurred 📍", 
        ['Residential areas', 'Office areas', 'Recreational areas', 'Industrial areas', 'Church areas', 'Market areas', 
        'Rural village areas', 'Outside rural areas', 'Hospital areas', 'School areas', 'Rural village areasOffice areas', 
        'Other'], index=0))

with col3:
    data.append(st.selectbox("Lanes or Medians 🚗", 
        ['One way', 'Undivided Two way', 'Two-way (divided with solid lines road marking)', 
        'Two-way (divided with broken lines road marking)', 'Double carriageway (median)', 'other'], 
        index=0))

    data.append(st.selectbox("Road Alignment 🛣️", 
        ['Tangent road with flat terrain', 'Tangent road with mild grade and flat terrain', 
        'Escarpments', 'Tangent road with rolling terrain', 'Gentle horizontal curve', 
        'Tangent road with mountainous terrain', 'Steep grade downward with mountainous terrain', 
        'Sharp reverse curve', 'Steep grade upward with mountainous terrain'], 
        index=0))

    data.append(st.selectbox("Types of Junction 🔀", 
        ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'T Shape', 'X Shape', 'Other'], 
        index=0))

    data.append(st.selectbox("Road Surface Type 🏎️", 
        ['Asphalt roads', 'Earth roads', 'Other', 'Asphalt roads with some distress', 'Gravel roads'], 
        index=0))

    data.append(st.selectbox("Road Surface Conditions 🌧️", 
        ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep'], 
        index=0))

    data.append(st.selectbox("Light Conditions 💡", 
        ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit'], 
        index=0))

with col4:

    data.append(st.selectbox("Weather Conditions ☁️", 
        ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy', 'Snow', 'Fog or mist'], index=0))

    data.append(st.selectbox("Type of Collision 🚗💥", 
        ['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision', 'Collision with roadside objects', 
        'Collision with animals', 'Other', 'Rollover', 'Fall from vehicles', 'Collision with pedestrians', 'With Train'], index=0))

    data.append(st.number_input("Number of Vehicles Involved 🚘", min_value=1, value=2))

    data.append(st.number_input("Number of Casualties ⚠️", min_value=0, value=2))

    data.append(st.selectbox("Vehicle Movement 🚦", 
        ['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go', 'Getting off', 'Reversing', 
        'Other', 'Parked', 'Stopping', 'Overtaking', 'Entering a junction'], index=0))

    data.append(st.selectbox("Cause of Accident ⚡", 
        ['Moving Backward', 'Overtaking', 'Changing lane to the left', 'Changing lane to the right', 'Overloading', 
        'Other', 'No priority to vehicle', 'No priority to pedestrian', 'No distancing', 'Getting off the vehicle improperly',
        'Improper parking', 'Overspeed', 'Driving carelessly', 'Driving to the left', 'Overturning', 'Turnover',
        'Driving under the influence of drugs', 'Drunk driving'], index=0))

st.markdown("---")

if st.button("Predict"):
    # Apply mapping for specific features
    data[3] = ages[data[3]]
    data[7] = exp[data[7]]
    
    for i in range(len(inx)):
        val=inx[i]
        data[val]=int(le_model[i].transform([data[val]])[0])
    
    prediction = int(rf_model.predict(np.array(data).reshape(1, -1))[0])
    
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(
            "<div style='background-color: #EAF8ED; color: black; padding: 10px; border-radius: 5px; text-align: center; width: 35%; margin: auto;'>"
            "<h2 style='color: green;'>✅ Slight Injury</h2></div>",
            unsafe_allow_html=True
        )
    elif prediction == 2:
        st.markdown(
            "<div style='background-color: #FFF4E5; color: black; padding: 10px; border-radius: 5px; text-align: center; width: 35%; margin: auto;'>"
            "<h2 style='color: orange;'>⚠ Serious Injury</h2></div>",
            unsafe_allow_html=True
        )
    elif prediction == 3:
        st.markdown(
            "<div style='background-color: #FCE8E6; color: black; padding: 10px; border-radius: 5px; text-align: center; width: 30%; margin: auto;'>"
            "<h2 style='color: red;'>🚨 Life Threatening Injury</h2></div>",
            unsafe_allow_html=True
        )

    
    st.markdown("</div>", unsafe_allow_html=True)