import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2  # Required for overlaying the heatmap on the original image
from tensorflow.keras.preprocessing import image

# TITLE DASHBOARD
st.markdown("<h1 class='center'>Skin Lesions Classification</h1>", unsafe_allow_html=True)

# Sidebar for information only
st.sidebar.title("Information:")
st.sidebar.markdown("**• Tugas Akhir Penerapan Transfer Learning InceptionResnetV2 Dalam Model Klasifikasi Penyakit Kulit Monkeypox Dengan Mempertimbangkan Hyperparameter Terbaik**")
st.sidebar.markdown("**• Nama: Luciano Rizky Pratama**")
st.sidebar.markdown("**• NIM: 24060121140156**")
st.sidebar.markdown("**• Dosen Pembimbing 1: Dr. Helmie Arif Wibawa, S.Si., M.Cs.**")
st.sidebar.markdown("**• Dosen Pembimbing 2: Rismiyati, B.Eng, M.Cs.**")

# Hide streamlit's footer and menu
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                            content:'This app is in its early stage. We recommend you to seek professional advice from a dermatologist. Thank you.'; 
                            visibility: visible;
                            display: block;
                            position: relative;
                            padding: 5px;
                            top: 2px;
                        }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add CSS styling for center alignment
st.markdown("""
    <style>
    .center {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the pre-trained model
model = tf.keras.models.load_model('InceptionResnetV2_fold_4.keras')

# Define the labels for the categories
labels = {
    0: 'Chickenpox',
    1: 'Cowpox',
    2: 'HFMD',
    3: 'Healthy',
    4: 'Measles',
    5: 'MPOX'
}

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to 224x224
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the image array (this is likely necessary)
    image_array = image_array / 255.0
    # Add an extra dimension to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to make predictions
def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make the prediction
    prediction = model.predict(processed_image)
    # Get the predicted label index
    label_index = np.argmax(prediction)
    # Get the predicted label
    predicted_label = labels[label_index]
    # Get the confidence level
    confidence = prediction[0][label_index] * 100
    return predicted_label, confidence

# Grad-CAM function
def grad_cam(img_array, model, last_conv_layer_name="conv_7b"):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    
    # Build the grad-cam model
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])
    
    # Record gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])  # Index of the top predicted class
        loss = predictions[:, predicted_class]
    
    # Compute the gradients
    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling
    
    # Create the Grad-CAM heatmap
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1).numpy()[0]
    
    # Normalize the heatmap
    cam = np.maximum(cam, 0)  # ReLU to keep only positive values
    cam = cam / np.max(cam)  # Normalize
    
    return cam

# Overlay the heatmap on the image
def overlay_heatmap(img, heatmap, alpha=0.5, cmap="jet"):
    # Ensure the heatmap is in the same size as the original image
    img_resized = np.array(img)  # Convert original image to numpy array
    heatmap_resized = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))  # Resize heatmap to match the original image dimensions
    
    # Scale heatmap to [0, 255] and apply the colormap
    heatmap = np.uint8(255 * heatmap_resized)  # Scale heatmap to [0, 255]
    heatmap = plt.cm.get_cmap(cmap)(heatmap)[:, :, :3]  # Convert to RGB format
    heatmap = np.uint8(heatmap * 255)  # Convert heatmap to uint8
    
    # Overlay the heatmap on the original image using weighted sum
    overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap, alpha, 0)
    return overlay

# Main app
def main():
    number = st.radio('Pick one', ['Upload from gallery', 'Capture by camera'])

    if number == 'Capture by camera':
        uploaded_file = st.camera_input("Take a picture")
    else:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg", "bmp"])

    # Display upload/capture interface
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert image for Grad-CAM
        img_array = preprocess_image(image)

        # Generate Grad-CAM heatmap
        cam = grad_cam(img_array, model)

        # Overlay the heatmap on the original image
        overlay_image = overlay_heatmap(image, cam, alpha=0.5)

        # Display the Grad-CAM overlay image first
        st.image(overlay_image, caption='Grad-CAM Overlay', use_container_width=True)
        
        # Process and classify the image
        predicted_label, confidence = predict(image)

        # Display the predicted label and confidence after the Grad-CAM image
        st.markdown("<h3 class='center'>Prediction:</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 class='center'>{predicted_label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p class='center'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
