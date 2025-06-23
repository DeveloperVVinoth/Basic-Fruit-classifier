import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model/fruit_knn_model.pkl")

# UI
st.title("Fruit Classifier 🍎🍊")
st.write("Enter fruit's weight and size to predict if it's an Apple or Orange.")

# Inputs
weight = st.number_input("Fruit Weight (grams):", min_value=100, max_value=300, step=1)
size = st.number_input("Fruit Size (diameter cm):", min_value=4.0, max_value=12.0, step=0.1)

# Predict
if st.button("Predict Fruit"):
    input_data = pd.DataFrame([[weight, size]], columns=["Weight", "Size"])
    prediction = model.predict(input_data)[0]
    st.success(f"This fruit is likely a **{prediction}** 🍏🍊")