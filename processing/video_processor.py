# processing/video_processor.py

import cv2
import pandas as pd
import tempfile
from datetime import datetime


from models.emotion_model import detect_emotion

from processing.preprocessing import (
    clean_bbox
)

from processing.engagement import (
    calculate_engagement
)

from config.setting import (
    FRAME_SKIP,
    CONFIDENCE_THRESHOLD,
    CLASS_NAMES,
    EMOTION_INTERVAL
)

from utils.logger import logger




def process_video(
        video_path,
        model,
        device,
        tracker
):


    start_time = datetime.now()


    cap = cv2.VideoCapture(
        video_path
    )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )



    output_file = tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".mp4"

    )



    writer = cv2.VideoWriter(

        output_file.name,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (width,height)

    )



    frame_id = 0


    results_data = []


    emotion_cache = {}



    while True:


        ret, frame = cap.read()


        if not ret:

            break



        if frame_id % FRAME_SKIP != 0:


            writer.write(frame)

            frame_id += 1

            continue



        try:


            detections=[]



            yolo_result = model.predict(

                frame,

                conf=
                CONFIDENCE_THRESHOLD,


                device=device,

                verbose=False

            )[0]



            for box in yolo_result.boxes:


                cls_id = int(
                    box.cls[0]
                )


                confidence = float(
                    box.conf[0]
                )



                x1,y1,x2,y2 = map(

                    int,

                    box.xyxy[0]

                )


                behavior = CLASS_NAMES[cls_id]



                detections.append(

                    [

                        [
                            x1,
                            y1,
                            x2-x1,
                            y2-y1
                        ],

                        confidence,

                        behavior

                    ]

                )



            tracks = tracker.update_tracks(

                detections,

                frame=frame

            )



            frame_engagement=[]



            for track in tracks:


                if not track.is_confirmed():

                    continue



                track_id = track.track_id



                x1,y1,x2,y2 = map(

                    int,

                    track.to_ltrb()

                )



                x1,y1,x2,y2 = clean_bbox(

                    x1,

                    y1,

                    x2,

                    y2,

                    width,

                    height

                )



                behavior = (

                    track.det_class

                    if track.det_class

                    else

                    "unknown"

                )



                if (

                    track_id in emotion_cache

                    and

                    frame_id % EMOTION_INTERVAL !=0

                ):


                    emotion = emotion_cache[track_id]


                else:


                    crop = frame[
                        y1:y2,
                        x1:x2
                    ]


                    emotion = detect_emotion(
                        crop
                    )


                    emotion_cache[track_id]=emotion




                engagement = calculate_engagement(

                    behavior,

                    emotion

                )



                status = engagement["status"]



                color = (

                    (0,255,0)

                    if status=="engaged"

                    else

                    (0,0,255)

                )



                cv2.rectangle(

                    frame,

                    (x1,y1),

                    (x2,y2),

                    color,

                    2

                )



                label = (

                    f"{behavior} | "
                    f"{emotion} | "
                    f"{engagement['score']}%"

                )



                cv2.putText(

                    frame,

                    label,

                    (x1,y1-10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    color,

                    2

                )



                frame_engagement.append(

                    status

                )



                results_data.append(

                    {

                    "frame":frame_id,

                    "student_id":track_id,

                    "behavior":behavior,

                    "emotion":emotion,

                    "score":engagement["score"],

                    "engagement":status

                    }

                )




            if frame_engagement:


                percentage = round(

                    (

                    frame_engagement.count(
                        "engaged"
                    )

                    /

                    len(frame_engagement)

                    )

                    *

                    100,

                    2

                )


                cv2.putText(

                    frame,

                    f"Class Engagement: {percentage}%",

                    (20,40),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1,

                    (255,255,0),

                    2

                )



            writer.write(frame)



        except Exception as e:


            logger.error(
                str(e)
            )



        frame_id += 1




    cap.release()

    writer.release()



    df = pd.DataFrame(
        results_data
    )


    processing_time = (

        datetime.now() - start_time

    )



    return (

        output_file.name,

        df,

        str(processing_time)

    )