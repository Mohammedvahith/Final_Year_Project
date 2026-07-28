import streamlit as st
import torch

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from config.setting import (
    HF_REPO_ID,
    HF_MODEL_NAME
)



@st.cache_resource
def load_yolo_model():


    model_path = hf_hub_download(

        repo_id=HF_REPO_ID,

        filename=HF_MODEL_NAME

    )


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model = YOLO(model_path)


    model.to(device)



    return model, device