import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from deepface import DeepFace
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import plotly.express as px

# ✅ Initialize session state
for key in ['processed', 'output_video', 'output_csv', 'df', 'process_time', 'start_processing']:
    if key not in st.session_state:
        st.session_state[key] = False if key in ['processed', 'start_processing'] else None

# ✅ Set Page config
st.set_page_config(page_title="Engagement Detection Dashboard", page_icon="🎥", layout="wide")

# ✅ Heading
st.title("🎯 Student Engagement Detection System")
st.markdown("Upload your **video** to detect behavior, emotion, and engagement.")

# ✅ Sidebar
st.sidebar.header("📂 Upload Video")
uploaded_video = st.sidebar.file_uploader("Upload Classroom Video", type=["mp4", "avi", "mov"])

# ✅ Model Load (Fixed inside code)
model_path = hf_hub_download(repo_id="Vahith1/yolov9-engagement", filename="best.pt")

# Load the YOLOv9 model
model = YOLO(model_path)

# ✅ DeepSORT Tracker
tracker = DeepSort(max_age=70, n_init=2, nms_max_overlap=0.5, max_cosine_distance=0.4)

# ✅ Behavior Classes
CLASS_NAMES = ['listening', 'reading', 'sleeping', 'turning', 'using_mobile', 'writing']

def get_engagement(behavior, emotion):
    behavior = behavior.lower()
    emotion = emotion.lower()
    engaged_behaviors = {"listening", "reading", "writing"}
    disengaged_behaviors = {"sleeping", "using_mobile", "turning"}

    if behavior in engaged_behaviors:
        
        return "engaged"
    if behavior in disengaged_behaviors:
        return "disengaged"

    positive_emotions = {"happy", "neutral", "focused"}
    return "engaged" if emotion in positive_emotions else "disengaged"

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

# ✅ Start button triggers processing
if st.button("🚀 Start Processing"):
    if uploaded_video:
        st.session_state.start_processing = True
        st.session_state.processed = False
    else:
        st.warning("Please upload a video before starting processing.")

# ✅ Process video only after button click
if st.session_state.start_processing and uploaded_video:
    with st.spinner('Processing... please wait ⏳'):
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(3)), int(cap.get(4))

        temp_output_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_output_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        out = cv2.VideoWriter(temp_output_video.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_id = 0
        engagement_data = []
        FRAME_SKIP = 4
        track_emotion_cache = {}
        start_time = datetime.now()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % FRAME_SKIP != 0:
                frame_id += 1
                continue

            results = model.predict(source=frame, conf=0.5, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                behavior = CLASS_NAMES[cls_id]
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / (height + 1e-5)
                if height < 50 or aspect_ratio > 2.5 or aspect_ratio < 0.2:
                    continue
                detections.append([[x1, y1, width, height], conf, behavior])

            final_detections = []
            used = [False] * len(detections)
            for i in range(len(detections)):
                if used[i]: continue
                x1, y1, w1, h1 = detections[i][0]
                box1 = (x1, y1, x1 + w1, y1 + h1)
                for j in range(i + 1, len(detections)):
                    if used[j]: continue
                    x2, y2, w2, h2 = detections[j][0]
                    box2 = (x2, y2, x2 + w2, y2 + h2)
                    if compute_iou(box1, box2) > 0.5:
                        used[j] = True
                final_detections.append(detections[i])

            tracks = tracker.update_tracks(final_detections, frame=frame)
            frame_engagements = []

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                behavior = track.det_class
                face_crop = frame[y1:y2, x1:x2]
                if track_id in track_emotion_cache and frame_id % (FRAME_SKIP * 3) != 0:
                    emotion = track_emotion_cache[track_id]
                else:
                    try:
                        analysis = DeepFace.analyze(face_crop, actions=["emotion"], enforce_detection=False, detector_backend='opencv')[0]
                        emotion = analysis["dominant_emotion"]
                        track_emotion_cache[track_id] = emotion
                    except:
                        emotion = "unknown"

                engagement = get_engagement(behavior, emotion)
                label = f"{behavior}, {emotion}, {engagement}"
                color = (0, 255, 0) if engagement == "engaged" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                frame_engagements.append(engagement)
                engagement_data.append({
                    "frame": frame_id,
                    "track_id": track_id,
                    "behavior": behavior,
                    "emotion": emotion,
                    "engagement": engagement
                })

            if frame_engagements:
                score = round((frame_engagements.count("engaged") / len(frame_engagements)) * 100, 2)
                overlay_text = f"Engagement: {score}%"
            else:
                overlay_text = "Engagement: No data"

            cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            out.write(frame)
            frame_id += 1

        cap.release()
        out.release()

        end_time = datetime.now()
        df = pd.DataFrame(engagement_data)
        df.to_csv(temp_output_csv.name, index=False)

        # ✅ Save outputs to session state
        st.session_state.output_video = temp_output_video.name
        st.session_state.output_csv = temp_output_csv.name
        st.session_state.df = df
        st.session_state.process_time = str(end_time - start_time)
        st.session_state.processed = True
        st.session_state.start_processing = False

        st.success(f"✅ Processing Completed (Time: {str(end_time - start_time)})")

# ✅ Display dashboard & downloads after processing
if st.session_state.processed:
    with open(st.session_state.output_video, 'rb') as f:
        st.download_button('⬇️ Download Processed Video', f, file_name='output_processed.mp4')

    with open(st.session_state.output_csv, 'rb') as f:
        st.download_button('⬇️ Download Engagement CSV', f, file_name='engagement_data.csv')

    df = st.session_state.df
    st.header("📊 Engagement Analytics Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Frames", df['frame'].nunique())
    with col2:
        st.metric("Unique Students", df['track_id'].nunique())
    with col3:
        avg_engagement = round((df['engagement'] == "engaged").mean() * 100, 2)
        st.metric("Avg Engagement", f"{avg_engagement}%")

    fig1 = px.histogram(df, x="behavior", color="engagement", barmode="group", title="Behavior vs Engagement")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="emotion", color="engagement", barmode="group", title="Emotion vs Engagement")
    st.plotly_chart(fig2, use_container_width=True)

    st.header("🎓 Adaptive Teaching Suggestions")
    if avg_engagement > 75:
        st.success("High Engagement ✅: Maintain current teaching pace and consider challenging tasks or peer discussions.")
    elif 50 < avg_engagement <= 75:
        st.warning("Moderate Engagement ⚠️: Include short interactive activities or refreshers.")
    else:
        st.error("Low Engagement ❌: Switch to visual aids, gamified learning, or discussion-based activities.")
