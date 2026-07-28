# app.py

import streamlit as st
import tempfile


from models.yolo_loader import load_yolo_model
from tracking.tracker import create_tracker
from processing.video_processor import process_video
from analytics.dashboard import show_dashboard


# =====================================================
# PAGE CONFIGURATION (must be first Streamlit command)
# =====================================================

st.set_page_config(

    page_title="Student Engagement AI",

    page_icon="🎓",

    layout="wide"

)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800;900&display=swap');

* {

    font-family: 'Space Grotesk', sans-serif !important;

}

/* Hide Streamlit header */
header[data-testid="stHeader"] {
    display: none;
}

/* Hide toolbar */
[data-testid="stToolbar"] {
    display: none;
}

/* Hide top decoration */
[data-testid="stDecoration"] {
    display: none;
}

/* Hide Deploy button */
[data-testid="stDeployButton"] {
    display: none;
}

/* Hide footer */
footer {
    display: none;
}

/* Hide Main Menu */
#MainMenu {
    visibility: hidden;
}

/* Remove top padding */
.block-container {
    padding-top: 3rem;
}

@keyframes slideLeft {

    from {

        opacity: 0;

        transform: translateX(-80px);

    }

    to {

        opacity: 1;

        transform: translateX(0);

    }

}


/* =========================
   GLOBAL BACKGROUND
========================= */

.stApp {

    background:
    linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #1e293b
    );

}


/* =========================
   PAGE LOAD ANIMATIONS
========================= */


@keyframes slideDown {


    from {

        opacity:0;

        transform:translateY(-50px);

    }


    to {

        opacity:1;

        transform:translateY(0);

    }

}



@keyframes fadeUp {


    from {

        opacity:0;

        transform:translateY(40px);

    }


    to {

        opacity:1;

        transform:translateY(0);

    }

}


/* =========================
   MAIN TITLE
========================= */


div.main-title {

    text-align:center !important;

    font-size:52px !important;

    line-height:1.2 !important;

    font-weight:900 !important;

    animation:slideDown 1s ease-out;

}


div.main-title p {

    font-size:52px !important;

    animation:
    fadeUp 1s ease-out;

}


.section-title {

    text-align:left;

    color:white;

    font-size:30px;

    font-weight:800;

    margin-top:20px;

    margin-bottom:25px;

    letter-spacing:1px;

    animation:slideDown 1s ease-out;

}

/* =========================
   SUBTITLE
========================= */


div.main-subtitle {

    text-align:center !important;

    font-size:22px !important;

    line-height:1.5 !important;

    animation:
    fadeUp 1s ease-out;

}


div.main-subtitle p {

    font-size:22px !important;

    animation:
    fadeUp 1s ease-out;

}


/* =========================
   PIPELINE CARDS
========================= */


.card {

    background:

    linear-gradient(
        145deg,
        rgba(255,255,255,0.12),
        rgba(255,255,255,0.04)
    );


    backdrop-filter:blur(18px);


    padding:30px;


    border-radius:24px;


    text-align:center;


    border:

    1px solid

    rgba(
        255,
        255,
        255,
        0.18
    );


    box-shadow:

    0px 15px 35px

    rgba(
        0,
        0,
        0,
        0.35
    );


    transition:all 0.3s ease;

    animation:
    fadeUp 1s ease-out;


}


/* =========================
   SIDEBAR
========================= */

[data-testid="stSidebar"] {

    animation: slideLeft 0.8s ease-out;

    background: linear-gradient(
        180deg,
        #0f172a,
        #1e293b
    );

    border-right: 1px solid rgba(255,255,255,0.08);

}


/* Hover effect */

.card:hover {


    transform:

    translateY(-10px);


    border:

    1px solid

    rgba(
        56,
        189,
        248,
        0.5
    );


    box-shadow:

    0px 20px 45px

    rgba(
        56,
        189,
        248,
        0.25
    );


}


/* Emoji */

.card-icon {


    font-size:45px;


}



/* Model name */

.card-value {


    font-size:30px;


    font-weight:800;


    color:white;


    margin-top:10px;


}



/* Description */


.card-title {


    color:#cbd5e1;


    font-size:17px;


    margin-top:8px;


}



/* Buttons */


.stDownloadButton button {

    width:100% !important;

    height:50px;

    border-radius:14px;

    font-size:16px;

    font-weight:700;

}



</style>

""",
unsafe_allow_html=True
)



# =====================================================
# SESSION STATE
# =====================================================

if "processed" not in st.session_state:

    st.session_state.processed = False


if "output_video" not in st.session_state:

    st.session_state.output_video = None


if "df" not in st.session_state:

    st.session_state.df = None


if "time" not in st.session_state:

    st.session_state.time = None



# =====================================================
# HEADER
# =====================================================

st.markdown(
"""
<div class="main-title">

🎓 Student Engagement AI

</div>


<div class="main-subtitle">

AI-powered classroom intelligence platform
for behavior, emotion and engagement analysis

</div>

""",
unsafe_allow_html=True
)


st.divider()



# =====================================================
# AI PIPELINE DISPLAY
# =====================================================

st.markdown(
"""
<div class="section-title">
🧠 AI Pipeline
</div>
""",
unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)



with col1:

    st.markdown(
    """
    <div class="card">

    <div class="card-icon">
    🎯
    </div>

    <div class="card-value">
    YOLO
    </div>

    <div class="card-title">
    Behavior Detection
    </div>

    </div>

    """,
    unsafe_allow_html=True
    )



with col2:

    st.markdown(
    """
    <div class="card">

    <div class="card-icon">
    👥
    </div>

    <div class="card-value">
    DeepSORT
    </div>

    <div class="card-title">
    Student Tracking
    </div>

    </div>

    """,
    unsafe_allow_html=True
    )



with col3:

    st.markdown(
    """
    <div class="card">

    <div class="card-icon">
    😊
    </div>

    <div class="card-value">
    DeepFace
    </div>

    <div class="card-title">
    Emotion Recognition
    </div>

    </div>

    """,
    unsafe_allow_html=True
    )



st.divider()



# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def get_models():

    from models.yolo_loader import load_yolo_model

    model, device = load_yolo_model()

    return model, device



# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Control Panel")


uploaded_video = st.sidebar.file_uploader(

    "📂 Upload Classroom Video",

    type=[
        "mp4",
        "avi",
        "mov"
    ]

)



if uploaded_video:

    st.sidebar.success(
        "Video uploaded successfully ✅"
    )



# =====================================================
# VIDEO PREVIEW
# =====================================================

if uploaded_video:

    st.markdown(
    """
    <div class="section-title">
    🎥 Input Video
    </div>
    """,
    unsafe_allow_html=True
    )
    
    st.video(
        uploaded_video
    )



# =====================================================
# PROCESS BUTTON
# =====================================================

if uploaded_video:


    if st.sidebar.button(
        "🚀 Start Analysis"
    ):

        with st.spinner("Loading AI models..."):

            model, device = get_models()

        progress = st.empty()


        progress.info(
            "🔍 AI analyzing classroom video..."
        )


        with st.spinner(
            "Processing video..."
        ):


            temp_video = tempfile.NamedTemporaryFile(

                delete=False,

                suffix=".mp4"

            )


            temp_video.write(

                uploaded_video.read()

            )


            video_path = temp_video.name



            tracker = create_tracker()



            output_video, df, processing_time = process_video(

                video_path,

                model,

                device,

                tracker

            )



            st.session_state.output_video = output_video

            st.session_state.df = df

            st.session_state.time = processing_time

            st.session_state.processed = True



        progress.success(
            "Analysis completed successfully ✅"
        )



# =====================================================
# RESULTS
# =====================================================

if st.session_state.processed:


    df = st.session_state.df


    st.divider()


    st.subheader(
        "📊 Classroom Overview"
    )


    total_students = df["student_id"].nunique()


    total_frames = df["frame"].nunique()



    avg_score = round(

        df["score"].mean(),

        2

    )



    col1,col2,col3,col4 = st.columns(4)



    with col1:

        st.markdown(
        f"""
        <div class="card">

        <div class="card-icon">
        👨‍🎓
        </div>

        <div class="card-value">
        {total_students}
        </div>

        <div class="card-title">
        Students
        </div>

        </div>
        """,
        unsafe_allow_html=True
        )



    with col2:

        st.markdown(
        f"""
        <div class="card">

        <div class="card-icon">
        🎞️
        </div>

        <div class="card-value">
        {total_frames}
        </div>

        <div class="card-title">
        Frames
        </div>

        </div>
        """,
        unsafe_allow_html=True
        )



    with col3:

        st.markdown(
        f"""
        <div class="card">

        <div class="card-icon">
        📈
        </div>

        <div class="card-value">
        {avg_score}%
        </div>

        <div class="card-title">
        Engagement
        </div>

        </div>
        """,
        unsafe_allow_html=True
        )



    with col4:

        st.markdown(
        f"""
        <div class="card">

        <div class="card-icon">
        ⏱️
        </div>

        <div class="card-value">
        {str(st.session_state.time).split(".")[0]}
        </div>

        <div class="card-title">
        Processing Time
        </div>

        </div>
        """,
        unsafe_allow_html=True
        )



    st.divider()

    # ===============================
    # DASHBOARD
    # ===============================


    show_dashboard(df)

    st.divider()

    # ===============================
    # DOWNLOADS
    # ===============================
    
    
    col1,col2 = st.columns(2)
    
    
    
    with col1:
    
        csv = df.to_csv(
            index=False
        )
    
    
        st.download_button(
    
            "⬇️ Download CSV",
    
            csv,
    
            "engagement_report.csv",
    
            "text/csv"
    
        )
    
    
    
    with col2:
    
    
        with open(
    
            st.session_state.output_video,
    
            "rb"
    
        ) as file:
    
    
            st.download_button(
    
                "⬇️ Download Processed Video",
    
                file,
    
                "processed_video.mp4"
    
            )
     