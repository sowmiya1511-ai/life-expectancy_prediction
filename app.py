import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Life Expectancy Prediction", layout="centered")
st.title("🌍 Life Expectancy Prediction App")

# Load required objects
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)   # THIS MUST BE A LIST OF COLUMN INDEXES

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.subheader("Enter Input Values")

def float_input(label):
    return st.number_input(label, value=0.0)

def int_input(label):
    return st.number_input(label, value=0)

# ----------------------------
# USER INPUT FIELDS
# ----------------------------

country = st.text_input("Country", "India")
status = st.selectbox("Status", ["Developing", "Developed"])
year = int_input("Year")
adult = float_input("Adult Mortality")
alcohol = float_input("Alcohol")
exp = float_input("Percentage Expenditure")
hep_b = float_input("Hepatitis B (%)")
measles = float_input("Measles Cases")
bmi = float_input("BMI")
under5 = float_input("Under-Five Deaths")
polio = float_input("Polio (%)")
diph = float_input("Diphtheria (%)")
hiv = float_input("HIV/AIDS (%)")
gdp = float_input("GDP")
thin5_9 = float_input("Thinness 5–9 years")
income = float_input("Income Composition of Resources")
schooling = float_input("Schooling")

# ----------------------------
# MAKE INPUT DATAFRAME
# ----------------------------

input_df = pd.DataFrame({
    "Country": [country],
    "Status": [status],
    "Year": [year],
    "Adult Mortality": [adult],
    "Alcohol": [alcohol],
    "percentage expenditure": [exp],
    "Hepatitis B": [hep_b],
    "Measles ": [measles],
    " BMI ": [bmi],
    "under-five deaths ": [under5],
    "Polio": [polio],
    "Diphtheria ": [diph],
    " HIV/AIDS": [hiv],
    "GDP": [gdp],
    " thinness 5-9 years": [thin5_9],
    "Income composition of resources": [income],
    "Schooling": [schooling]
})

# ----------------------------
# PREDICT BUTTON
# ----------------------------

if st.button("Predict Life Expectancy"):
    try:
        # Ensure all required columns exist
        required_cols = preprocessor.feature_names_in_
        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns in the correct order
        input_df = input_df[required_cols]

        # Apply preprocessing
        X_prep = preprocessor.transform(input_df)

        # Convert to DataFrame WITHOUT feature names
        X_prep_df = pd.DataFrame(X_prep)

        # Select only the chosen feature indexes
        X_final = X_prep_df.iloc[:, selected_features]

        # Predict
        pred = model.predict(X_final)[0]

        st.success(f"Predicted Life Expectancy: **{pred:.2f} years**")

    except Exception as e:
        st.error("Error occurred.")
        st.write(str(e))
