# app.py
import streamlit as st
import tempfile
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

st.set_page_config(layout="wide", page_title="AI Video Analytics")

st.title("🎥 AI Video Analytics Dashboard")
st.markdown(
    """
    Upload a video, inspect metadata (resolution, fps, frame count), preview frames,
    and optionally run AI object detection on frames.
    """
)

# ---------------------- Utility Functions ----------------------

@st.cache_data
def save_uploaded_video(uploaded_file):
    """Save uploaded file to a temp file and return its path."""
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    return tmp.name

def get_video_metadata(path):
    """Extract video metadata without caching cv2.VideoCapture object."""
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else None
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }

def read_frame_at(path, frame_number):
    """Return a PIL.Image for a specific frame index (0-based)."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def draw_boxes_on_image(pil_img, boxes, labels=None, scores=None):
    """Draw bounding boxes on PIL image."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], width=3, outline="red")
        caption = ""
        if labels and i < len(labels):
            caption += str(labels[i])
        if scores and i < len(scores):
            caption += f" {scores[i]:.2f}"
        if caption:
            draw.text((x1, y1 - 10), caption, fill="yellow", font=font)
    return img

# ---------------------- AI Model Loader (Optional) ----------------------

@st.cache_resource
def load_yolov5_model():
    """Try loading YOLOv5 model using torch.hub. Return None if fails."""
    try:
        import torch
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        model.eval()
        return model
    except Exception as e:
        st.session_state["model_error"] = str(e)
        return None

def run_detection_on_pil(model, pil_img):
    results = model(pil_img)
    df = results.pandas().xyxy[0]
    boxes, labels, scores = [], [], []
    for _, row in df.iterrows():
        boxes.append((row["xmin"], row["ymin"], row["xmax"], row["ymax"]))
        labels.append(row["name"])
        scores.append(row["confidence"])
    return boxes, labels, scores

# ---------------------- UI Layout ----------------------

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Steps")
    st.markdown(
        """
        1. 📥 Upload a video  
        2. 🎞️ View metadata  
        3. 🔍 Preview frames  
        4. 🤖 (Optional) Run AI object detection
        """
    )
    uploaded = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mkv"])
    st.markdown("---")
    st.checkbox("Enable AI Object Detection", key="enable_ai")

with col2:
    st.header("Preview / Results")

    if not uploaded:
        st.info("Upload a video to start.")
        st.stop()

    tmp_video_path = save_uploaded_video(uploaded)

    # Show video
    st.subheader("Video Player")
    st.video(tmp_video_path)

    # Metadata
    meta = get_video_metadata(tmp_video_path)
    st.subheader("Video Metadata")
    cols = st.columns(4)
    cols[0].metric("Resolution", f"{meta['width']}x{meta['height']}")
    cols[1].metric("FPS", f"{meta['fps']:.2f}")
    cols[2].metric("Frames", str(meta['frame_count']))
    dur = f"{meta['duration_sec']:.2f}s" if meta['duration_sec'] else "Unknown"
    cols[3].metric("Duration", dur)

    # Frame selection
    st.subheader("Frame Preview")
    fc = meta["frame_count"] or 1
    frame_idx = st.slider("Select frame", 0, max(fc - 1, 0), 0)
    pil_frame = read_frame_at(tmp_video_path, frame_idx)
    if pil_frame:
        st.image(pil_frame, caption=f"Frame {frame_idx}", use_container_width=True)
    else:
        st.error("Unable to read frame.")
        st.stop()

    # AI detection
    if st.session_state.enable_ai:
        with st.spinner("Loading AI model..."):
            model = load_yolov5_model()
        if not model:
            st.error("YOLOv5 model failed to load.")
            if "model_error" in st.session_state:
                st.text(st.session_state["model_error"])
        else:
            if st.button("Run Detection"):
                with st.spinner("Detecting..."):
                    boxes, labels, scores = run_detection_on_pil(model, pil_frame)
                    if boxes:
                        detected_img = draw_boxes_on_image(pil_frame, boxes, labels, scores)
                        st.image(detected_img, caption="Detections", use_container_width=True)
                        st.table(
                            {"Label": labels, "Score": [f"{s:.2f}" for s in scores]}
                        )
                    else:
                        st.info("No objects detected.")
