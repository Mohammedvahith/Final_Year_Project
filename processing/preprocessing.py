# processing/preprocessing.py


import cv2



def resize_frame(
        frame,
        width=960
):


    height = int(

        frame.shape[0]
        *
        width
        /
        frame.shape[1]

    )


    return cv2.resize(

        frame,

        (width, height)

    )





def clean_bbox(
        x1,
        y1,
        x2,
        y2,
        frame_width,
        frame_height
):


    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )


    x2 = min(
        frame_width,
        x2
    )


    y2 = min(
        frame_height,
        y2
    )


    return (
        x1,
        y1,
        x2,
        y2
    )