import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Life Expectancy Prediction", layout="centered")
st.title("🌍 Life Expectancy Prediction App")

# Load models
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)   # THIS IS A LIST OF FEATURE NAMES

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
# PREDICTION
# ----------------------------

if st.button("Predict Life Expectancy"):
    try:
        # Add missing columns
        required_cols = preprocessor.feature_names_in_
        for col in required_cols:
            if col not in input_df:
                input_df[col] = 0

        input_df = input_df[required_cols]

        # Preprocess
        X_prep = preprocessor.transform(input_df)

        # Build output column names manually
        new_cols = []

        num_cols = preprocessor.transformers_[0][2]
        cat_cols = preprocessor.transformers_[1][2]

        # numerical names (unchanged)
        new_cols.extend(["num__" + c for c in num_cols])

        # categorical names (after one-hot encoding)
        ohe = preprocessor.named_transformers_["cat"]
        ohe_cols = ohe.get_feature_names_out(cat_cols)
        new_cols.extend(ohe_cols)

        # Convert to DataFrame
        X_prep_df = pd.DataFrame(X_prep, columns=new_cols)

        # Select only your chosen features
        X_final = X_prep_df[selected_features]

        # Predict
        pred = model.predict(X_final)[0]

        st.success(f"Predicted Life Expectancy: **{pred:.2f} years**")

    except Exception as e:
        st.error("Error occurred.")
        st.write(str(e))
