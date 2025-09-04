import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np

# Title of the Dashboard
st.title("🎥 AI Video Analytics Dashboard")
st.write("Upload a video to analyze objects using AI.")

# Load pre-trained model (YOLOv5)
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded video
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Open video
    cap = cv2.VideoCapture("temp_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Show video info
    st.subheader("📹 Video Information")
    st.text(f"Resolution: {width}x{height}")
    st.text(f"Frame Rate: {fps} fps")
    st.text(f"Total Frames: {total_frames}")

    # Read first frame
    ret, frame = cap.read()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        st.subheader("🖼️ First Frame Preview")
        st.image(pil_img, width=600)

        # Run AI inference
        if st.button("🔍 Detect Objects with AI"):
            with st.spinner("Analyzing..."):
                results = model(pil_img)
                st.subheader("🎯 AI Detection Results")
                st.image(np.array(results.render()[0]), caption="Detected Objects", width=600)
                
                # Show detected labels
                labels = results.pandas().xyxy[0]['name'].tolist()
                st.write("**Detected Objects:**", ", ".join(set(labels)))

    cap.release()
else:
    st.info("Please upload a video file to begin.")