import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Downloading the model from the Model Hub
model_path = hf_hub_download(repo_id="skalpitin/Tourism-Package-Prediction", filename="Tourism-Package-Prediction_v1.joblib")

# Loading the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("`Visit with us` Prediction App")
st.write("The `Visit with us` Prediction App is an internal tool for the internal staff that predicts whether customers are going to take the tourism package or not.")
st.write("Kindly enter the customer details to check whether they are buy the tourism package.")

# Collecting user input
Age = st.number_input("Age (customer's age in years)", min_value=18, value=25)
TypeofContact = st.selectbox("Type of contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Large Business", "Small Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=0, max_value=25, value=1)
PreferredPropertyStar = st.selectbox("Preferred Property Star", ["1", "2", "3","4", "5"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=0, value=5)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number Of Children (Below 5) Visiting", min_value=0, value=2)
Designation = st.selectbox("Designation", ["Executive", "Manager", "AVP", "Senior Manager", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", ["1", "2", "3","4", "5"]) 
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
NumberOfFollowups = st.selectbox("Number Of Followups", ["1", "2", "3","4", "5"]) 
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=0.0, value=30.0)

# Converting inputs to a dataframe to pass to the model
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

# Setting the classification threshold
classification_threshold = 0.45

# Predict button -  Calling the model with input dataframe
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "take" if prediction == 1 else "not take"
    st.write(f"Based on the information provided, the customer is likely to {result} the product.")
