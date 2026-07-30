import os


# ==========================
# Hugging Face Model
# ==========================

HF_REPO_ID = "Vahith1/yolov9-engagement"
HF_MODEL_NAME = "best.pt"



# ==========================
# Detection Settings
# ==========================

CONFIDENCE_THRESHOLD = 0.65


CLASS_NAMES = [
    "listening",
    "reading",
    "sleeping",
    "turning",
    "using_mobile",
    "writing"
]



# ==========================
# Video Settings
# ==========================

FRAME_SKIP = 3


# ==========================
# Emotion Settings
# ==========================

EMOTION_INTERVAL = 120



# ==========================
# DeepSORT Settings
# ==========================

MAX_AGE = 200
N_INIT = 5
MAX_COSINE_DISTANCE = 0.5



# ==========================
# Output
# ==========================

OUTPUT_DIR = "outputs"


VIDEO_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "videos"
)


CSV_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "csv"
)


os.makedirs(
    VIDEO_OUTPUT_DIR,
    exist_ok=True
)


os.makedirs(
    CSV_OUTPUT_DIR,
    exist_ok=True
)