import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Life Expectancy Prediction", layout="centered")
st.title("🌍 Life Expectancy Prediction App")

# ----------------------------
# Load models and preprocessor
# ----------------------------
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("selected_features.pkl", "rb") as f:
    selected_features = pickle.load(f)

# Fix TargetEncoder categorical feature names
selected_features = [f.replace("cat__", "") for f in selected_features]

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

st.subheader("Enter Input Values")

# ----------------------------
# Input helper functions
# ----------------------------
def float_input(label):
    return st.number_input(label, value=0.0)

def int_input(label):
    return st.number_input(label, value=0)

# ----------------------------
# User input fields
# ----------------------------
inputs = {
    "Country": st.text_input("Country", "India"),
    "Status": st.selectbox("Status", ["Developing", "Developed"]),
    "Year": int_input("Year"),
    "Adult Mortality": float_input("Adult Mortality"),
    "Alcohol": float_input("Alcohol"),
    "percentage expenditure": float_input("Percentage Expenditure"),
    "Hepatitis B": float_input("Hepatitis B (%)"),
    "Measles ": float_input("Measles Cases"),
    " BMI ": float_input("BMI"),
    "under-five deaths ": float_input("Under-Five Deaths"),
    "Polio": float_input("Polio (%)"),
    "Diphtheria ": float_input("Diphtheria (%)"),
    " HIV/AIDS": float_input("HIV/AIDS (%)"),
    "GDP": float_input("GDP"),
    " thinness 5-9 years": float_input("Thinness 5–9 years"),
    "Income composition of resources": float_input("Income Composition of Resources"),
    "Schooling": float_input("Schooling")
}

input_df = pd.DataFrame([inputs])

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Life Expectancy"):
    try:
        # Ensure all required columns exist
        required_cols = preprocessor.feature_names_in_
        for col in required_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[required_cols]

        # Preprocess input
        X_prep = preprocessor.transform(input_df)

        # Build column names
        num_cols = preprocessor.transformers_[0][2]  # numeric columns
        num_feature_names = ["num__" + c for c in num_cols]

        cat_cols = preprocessor.transformers_[1][2]  # categorical columns
        cat_feature_names = list(cat_cols)  # TargetEncoder keeps original names

        all_feature_names = num_feature_names + cat_feature_names

        X_prep_df = pd.DataFrame(X_prep, columns=all_feature_names)

        # Select only features used by the model
        X_final = X_prep_df[selected_features]

        # Predict
        pred = model.predict(X_final)[0]

        st.success(f"Predicted Life Expectancy: **{pred:.2f} years**")

    except Exception as e:
        st.error("Error occurred during prediction.")
        st.write(str(e))
