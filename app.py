import streamlit as st
import joblib
import numpy as np

st.markdown(
    """
    <h1 style='color: navy;'>Water Purity Prediction</h1>
    <h2 style='color: navy;'>Enter the features to predict water purity</h2>
    <style>
    .stApp {
        background-color: #a2c4e6;  /* soft blue */
    }
    .stNumberInput input {
        background-color:#d7e9f7;   /* water theme blue */
    }
    .stNumberInput label {
        color: navy; 
        }
    .stButton > button {        
        background-color: #d7e9f7;  /* water theme blue */
        color: navy; 

    }
    .stText{
    color: navy;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# Load the trained model
model = joblib.load('D:\water_Purity\water_purity_model.pkl')


#st.title ("Water Purity Prediction")           

#st.write("""
    ### Enter the features to predict water purity
    #""")

# Input fields for the features required by your model
   									
ph = st.number_input("**PH**")
Hardness = st.number_input("**Hardness**")
Solids = st.number_input("**Solids**")
Chloramines = st.number_input("**Chloramines** ")
Sulphate = st.number_input("**Sulphate**")
Conductivity = st.number_input("**Conductivity**")
Organic_carbon = st.number_input("**Organic carbon**")
Trihalomethanes = st.number_input("**Trihalomethanes** ")
Turbidity = st.number_input("**Turbidity**")
    

    # When the user clicks the predict button
if st.button("Predict"):
    if any(v == 0.0 for v in [ph, Hardness, Solids, Chloramines, Sulphate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]):
        st.write("Please fill in all the fields.")
    else:
        features = np.array([[ph,Hardness,Solids,Chloramines,Sulphate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity]])  # Adjust according to your model's input shape
        prediction = model.predict(features)
        st.write("Prediction:","Water is safe to drink" if prediction ==1 else "water is not safe to drink")
    


