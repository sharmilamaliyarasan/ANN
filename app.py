import streamlit as st
import numpy as np
import pickle

# Import scikit-learn classes so pickle can find them
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# Load models
# -------------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌸 Iris ANN Classifier", layout="centered")
st.title("🌸 Iris Flower Classifier using ANN")
st.write("Enter flower measurements and predict the species!")

sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 0.1)
sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 0.1)
petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 0.1)
petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.1)

if st.button("🔍 Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)  # scale inputs
    pred = model.predict(features_scaled)         # predict
    species = encoder.inverse_transform(pred)[0]  # decode label

    st.success(f"🌼 Predicted Species: **{species}**")
