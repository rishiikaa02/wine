import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Wine Quality App", layout="centered")

# ------------------ TITLE ------------------
st.title("🍷 Wine Quality Prediction App")
st.write("Enter chemical properties to predict wine quality.")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("wine.csv")

# 🔥 Remove unwanted column if present
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# ------------------ TRAIN MODEL ------------------
X = df.drop("quality", axis=1)
y = df["quality"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

st.success("Model loaded successfully!")

# ------------------ INPUT FIELDS ------------------
fixed_acidity = st.number_input("Fixed Acidity", value=0.0, step=0.01)
volatile_acidity = st.number_input("Volatile Acidity", value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", value=0.0, step=0.01)
chlorides = st.number_input("Chlorides", value=0.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=0.0, step=1.0)
density = st.number_input("Density", value=0.0, step=0.001)
pH = st.number_input("pH", value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", value=0.0, step=0.01)

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

    # Ensure exact feature match
    input_data = pd.DataFrame(input_values, columns=X.columns)

    prediction = model.predict(input_data)[0]

    # 🔥 FINAL OUTPUT (decimal as you wanted)
    st.success(f"🍷 Wine Quality: {round(prediction, 2)}")

    # ------------------ INTERPRETATION ------------------
    if prediction >= 7:
        st.success("✨ Good Quality Wine")
    elif prediction >= 5:
        st.warning("🙂 Average Quality Wine")
    else:
        st.error("⚠️ Poor Quality Wine")
