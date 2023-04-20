import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import streamlit as st

def get_dominant_colors(image, k=5):
    # Load image and convert to RGB format
    image = Image.open(image).convert('RGB')
    
    # Reshape image into 2D array of pixels
    pixels = np.array(image).reshape(-1, 3)
    
    # Fit KMeans clustering model to pixel values
    kmeans = KMeans(n_clusters=k).fit(pixels)
    
    # Get centroids and counts for each cluster
    centroids = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    
    # Compute percentages of pixels for each dominant color
    percentages = counts / sum(counts) * 100
    
    return centroids, percentages

def show_dominant_colors(centroids, percentages):
    # Create color boxes with corresponding RGB values
    color_boxes = [f"<div style='background-color: rgb{tuple(map(int, color))}; width: 50px; height: 50px; display: inline-block;'></div>" for color in centroids]
    
    # Combine color boxes and percentages into a single string
    colors_and_percentages = [f"{color_box} {percent:.1f}%" for color_box, percent in zip(color_boxes, percentages)]
    colors_and_percentages_str = "<br>".join(colors_and_percentages)
    
    # Display color boxes and percentages in Streamlit app
    st.markdown(colors_and_percentages_str, unsafe_allow_html=True)

# Create Streamlit app
st.title("Dominant Colors in Image")

# Upload image and select number of dominant colors to find
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
k = st.slider("Number of dominant colors", min_value=1, max_value=10, value=5)

# When an image is uploaded, find the dominant colors and display them in the app
if uploaded_file is not None:
    centroids, percentages = get_dominant_colors(uploaded_file, k)
    show_dominant_colors(centroids, percentages)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

