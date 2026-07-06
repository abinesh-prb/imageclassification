import streamlit as st
import joblib
import PIL.Image as Image
import numpy as np

st.title("😊 Smile Detection")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
metrics = joblib.load("metrics.pkl")



uploaded_file=st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,width=300)
    grayscale=image.convert('L')
    resized_image=grayscale.resize((64,64))

    image_array=np.array(resized_image).flatten().reshape(1,-1)
    image_array=scaler.transform(image_array)

    prediction=model.predict(image_array)
    probability = model.predict_proba(image_array)


    if prediction[0] == 1:
      st.success("😊 Smiling Face")
    else:
      st.error("😐 Non-Smiling Face")

    st.write("Prediction Probability")
    st.write(probability)
    st.subheader("Model Performance")

    st.write("Accuracy")
    st.write(metrics["accuracy"])

    st.subheader("Confusion Matrix")
    st.write(metrics["confusion_matrix"])

    st.subheader("Classification Report")
    st.text(metrics["classification_report"])



