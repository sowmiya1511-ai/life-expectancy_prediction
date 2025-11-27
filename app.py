import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Life Expectancy Prediction", layout="centered")
st.title("🌍 Life Expectancy Prediction App")

# Load required objects
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.subheader("Enter Input Values")

def float_input(label):
    return st.number_input(label, value=0.0)

def int_input(label):
    return st.number_input(label, value=0)

# Only selected features
country = st.text_input("Country", "India")                         # cat__Country
status = st.selectbox("Status", ["Developing", "Developed"])       # cat__Status
year = int_input("Year")                                           # num__Year
adult = float_input("Adult Mortality")                             # num__Adult Mortality
alcohol = float_input("Alcohol")                                   # num__Alcohol
exp = float_input("Percentage Expenditure")                        # num__percentage expenditure
hep_b = float_input("Hepatitis B (%)")                             # num__Hepatitis B
measles = float_input("Measles Cases")                             # num__Measles 
bmi = float_input("BMI")                                           # num__ BMI 
under5 = float_input("Under-Five Deaths")                          # num__under-five deaths 
polio = float_input("Polio (%)")                                   # num__Polio
diph = float_input("Diphtheria (%)")                               # num__Diphtheria 
hiv = float_input("HIV/AIDS (%)")                                  # num__ HIV/AIDS
gdp = float_input("GDP")                                           # num__GDP
thin5_9 = float_input("Thinness 5–9 years")                        # num__ thinness 5-9 years
income = float_input("Income Composition of Resources")            # num__Income composition of resources
schooling = float_input("Schooling")                               # num__Schooling


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

if st.button("Predict Life Expectancy"):
    try:
        # Ensure all required columns exist
        required_cols = preprocessor.feature_names_in_
        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0  # missing columns filled automatically

        # Reorder columns exactly as preprocessor expects
        input_df = input_df[required_cols]

        # Preprocess
        X_prep = preprocessor.transform(input_df)

        # Select features
        X_final = pd.DataFrame(X_prep, columns=preprocessor.get_feature_names_out())
        X_final = X_final[selected_features]

        # Predict
        pred = model.predict(X_final)[0]

        st.success(f"Predicted Life Expectancy: **{pred:.2f} years**")

    except Exception as e:
        st.error("Error occurred.")
        st.write(e)

