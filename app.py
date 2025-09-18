import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="🌸 Iris ANN Classifier", layout="centered")

st.title("🌸 Iris Flower Classifier using ANN")
st.write("Enter flower measurements and predict the species!")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

if st.button("🔍 Predict"):
   
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)
    species = encoder.inverse_transform(pred)[0]

    st.success(f"🌼 Predicted Species: **{species}**")
