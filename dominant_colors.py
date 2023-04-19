import cv2
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st

# Define a function to get the dominant colors and their percentages
def get_dominant_colors(image, k=3):
    # Reshape the image into a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    
    # Create a KMeans object and fit it to the pixel values
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixel_values)
    
    # Get the centroids and convert them back to the 8-bit color space
    centroids = kmeans.cluster_centers_
    centroids = np.uint8(centroids)
    
    # Get the number of pixels assigned to each centroid
    counts = np.bincount(kmeans.labels_)
    
    # Compute the percentage of pixels for each dominant color
    percentages = counts / len(pixel_values) * 100
    
    # Return the dominant colors and their percentages
    return centroids, percentages

# Define a function to display the dominant colors and their percentages
def show_dominant_colors(colors, percentages):
    st.write("**Dominant Colors:**")
    for color, percentage in zip(colors, percentages):
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        st.write(f"<div style='background-color:{hex_color}; width:50px; height:50px; display:inline-block;'></div>", unsafe_allow_html=True)
        st.write(f"{percentage:.2f}%")

# Create a Streamlit app
st.title("Dominant Colors Finder")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    # Load the image using OpenCV
    image = cv2.imread(uploaded_file.name)
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the dominant colors and their percentages
    k = st.slider("Number of Colors", min_value=2, max_value=10, value=3, step=1)
    colors, percentages = get_dominant_colors(image, k=k)
    
    # Display the original image and the dominant colors with their percentages
    st.image(image, caption="Original Image", use_column_width=True)
    show_dominant_colors(colors, percentages)
