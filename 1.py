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

# ------------------ TRAIN MODEL ------------------
X = df.drop('quality', axis=1)
y = df['quality']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

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

    # IMPORTANT: Use DataFrame with exact column names
    input_data = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Output
    st.success(f"🍷 Predicted Wine Quality: {prediction}")

    # Optional: Better interpretation
    if prediction >= 7:
        st.success("✨ Good Quality Wine")
    elif prediction >= 5:
        st.warning("🙂 Average Quality Wine")
    else:
        st.error("⚠️ Poor Quality Wine")
