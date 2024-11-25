import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('path_to_your_saved_model.h5')

# Define the class labels
class_labels = ['acne', 'atopic', 'bcc']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img

# Function to make prediction
def predict_disease(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    predicted_probability = prediction[0, predicted_class_index]
    return predicted_class, predicted_probability

# Streamlit app
def main():
    st.title('Skin Disease Prediction')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        disease, probability = predict_disease(img)
        st.write(f"Prediction: {disease} (Probability: {probability:.2f})")

if __name__ == '__main__':
    main()
