import streamlit as st
from deepface import DeepFace



@st.cache_resource
def load_emotion_model():

    return True



def detect_emotion(face):

    try:

        result = DeepFace.analyze(

            face,

            actions=[
                "emotion"
            ],

            enforce_detection=False,

            detector_backend="retinaface"

        )


        emotion = result[0]["dominant_emotion"]


        return emotion



    except Exception:

        return "unknown"