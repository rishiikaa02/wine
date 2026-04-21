import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Wine Quality App", layout="centered")

# ------------------ TITLE ------------------
st.title("🍷 Wine Quality Prediction App")
st.write("Enter chemical properties to predict wine quality.")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("wine.csv")

# 🔥 FIX: remove Id column
X = df.drop(['quality', 'Id'], axis=1)
y = df['quality']

# ------------------ TRAIN MODEL ------------------
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

st.success("Model loaded successfully!")

# ------------------ INPUT FIELDS ------------------
fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.number_input("Residual Sugar", 0.5, 15.0, 2.0)
chlorides = st.number_input("Chlorides", 0.01, 0.2, 0.05)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 1.0, 80.0, 15.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 5.0, 300.0, 50.0)
density = st.number_input("Density", 0.990, 1.005, 0.996)
pH = st.number_input("pH", 2.5, 4.5, 3.2)
sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.6)
alcohol = st.number_input("Alcohol", 8.0, 15.0, 10.0)

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

    # Match training columns
    input_data = pd.DataFrame(input_values, columns=X.columns)

    prediction = int(model.predict(input_data)[0])

    st.success(f"🍷 Predicted Wine Quality: {prediction}")

    if prediction >= 7:
        st.success("✨ Good Quality Wine")
    elif prediction >= 5:
        st.warning("🙂 Average Quality Wine")
    else:
        st.error("⚠️ Poor Quality Wine")
