# 🔍 Iris ANN Classification Dashboard 
## App Link: 
https://ls7mg7pg9jmthtdmhwdphu.streamlit.app/
## 📌 Project Overview

This project is an End-to-End Machine Learning application that classifies Iris flower species using an Artificial Neural Network (ANN).

It provides a Gradio-powered interactive dashboard where users can input sepal and petal measurements, predict the species, and visualize training performance.

It covers the complete ML lifecycle:

## 📂 Data Preprocessing – Encode target labels, scale input features with StandardScaler

⚙️ ANN Modeling – Multi-layer Perceptron (MLPClassifier) with tunable hyperparameters

📊 Evaluation Metrics – Accuracy, training loss curve

📈 Visualization – Loss convergence plot and optional accuracy curve


## 🚀 Features

✅ ANN-based classification of Iris species (Iris-setosa, Iris-versicolor, Iris-virginica)

✅ Hyperparameter tuning via GridSearchCV (hidden layers, activation, solver, learning rate)

✅ Feature scaling and label encoding for robust training

✅ Save/load trained model, scaler, and label encoder for predictions

✅ Interactive Gradio web app for live predictions

✅ Visualization of training loss and model convergence

## 🛠️ Tech Stack

Python 🐍

Scikit-learn → ANN, GridSearchCV, metrics

Pandas / NumPy → Data handling

Matplotlib / Seaborn → Visualization

Joblib → Model persistence

Gradio → Interactive dashboard frontend

## 📂 Project Structure

Iris_ANN_Dashboard/

├── app.py                 
├── train.py             
├── iris_ann_model.pkl    
├── scaler.pkl             
├── label_encoder.pkl     
├── requirements.txt
└── README.md


## ⚙️ Installation & Setup

1️⃣ Clone the repository

2️⃣ Install dependencies:

pip install -r requirements.txt


## 3️⃣ Run the Gradio app:

python app.py


## 📊 Example Workflow

Preprocess dataset → Encode labels & scale features

Train baseline ANN → Evaluate test accuracy

Run GridSearchCV → Find best hidden layers, activation, solver, learning rate

Evaluate tuned model → Accuracy, loss curve

Save trained objects → ANN model, scaler, label encoder

Open dashboard → Input sample sepal/petal values → Predict species

## 📊 Evaluation Metrics

Accuracy → Baseline: ~95%, Tuned model: 100%

Loss Curve → Training loss convergence per iteration

Optional Accuracy Curve → Training vs testing accuracy

## 🎯 Future Enhancements

Upload custom datasets for predictions

Visualize feature importance & decision boundaries

Export prediction reports (PDF/CSV)

Role-based access (Admin vs User)

Deploy on Hugging Face / Streamlit for public access
