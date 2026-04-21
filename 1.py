import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ------------------ LOAD DATA ------------------
df = pd.read_csv("wine.csv")

# ------------------ TRAIN MODEL ------------------
X = df.drop('quality', axis=1)
y = df['quality']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# ------------------ UI ------------------
st.set_page_config(page_title="Wine Quality App", layout="centered")

st.title("🍷 Wine Quality Prediction App")
st.write("Enter chemical properties to predict wine quality.")

st.success("Model loaded successfully!")

# ------------------ INPUT FIELDS ------------------
fixed_acidity = st.number_input("Fixed Acidity", 0.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0)
citric_acid = st.number_input("Citric Acid", 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0)
chlorides = st.number_input("Chlorides", 0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0)
density = st.number_input("Density", 0.0)
pH = st.number_input("pH", 0.0)
sulphates = st.number_input("Sulphates", 0.0)
alcohol = st.number_input("Alcohol", 0.0)

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict Quality"):
    
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"🍷 Predicted Wine Quality: {prediction}")
