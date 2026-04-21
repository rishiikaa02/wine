import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # for decimal output

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Wine Quality App", layout="centered")

# ------------------ TITLE ------------------
st.title("🍷 Wine Quality Prediction App")
st.write("Enter chemical properties to predict wine quality.")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("wine.csv")

# 🔥 REMOVE unwanted column if exists
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# ------------------ TRAIN MODEL ------------------
X = df.drop("quality", axis=1)
y = df["quality"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

st.success("Model loaded successfully!")

# ------------------ INPUT FUNCTION ------------------
def get_input(label):
    return float(st.text_input(label, "0"))

# ------------------ INPUT FIELDS ------------------
fixed_acidity = get_input("Fixed Acidity")
volatile_acidity = get_input("Volatile Acidity")
citric_acid = get_input("Citric Acid")
residual_sugar = get_input("Residual Sugar")
chlorides = get_input("Chlorides")
free_sulfur_dioxide = get_input("Free Sulfur Dioxide")
total_sulfur_dioxide = get_input("Total Sulfur Dioxide")
density = get_input("Density")
pH = get_input("pH")
sulphates = get_input("Sulphates")
alcohol = get_input("Alcohol")

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Quality"):

    input_values = [[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]]

    # Ensure correct structure
    input_data = pd.DataFrame(input_values, columns=X.columns)

    prediction = model.predict(input_data)[0]

    # 🔥 FINAL OUTPUT (your requirement)
    st.success(f"🍷 Wine Quality: {round(prediction, 2)}")

    # Optional label
    if prediction >= 7:
        st.success("✨ Good Quality Wine")
    elif prediction >= 5:
        st.warning("🙂 Average Quality Wine")
    else:
        st.error("⚠️ Poor Quality Wine")
