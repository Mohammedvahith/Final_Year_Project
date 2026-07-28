# processing/engagement.py


BEHAVIOR_SCORE = {

    "listening": 0.90,

    "writing": 0.85,

    "reading": 0.75,

    "turning": 0.40,

    "using_mobile": 0.15,

    "sleeping": 0.05

}



EMOTION_SCORE = {

    "happy": 0.90,

    "neutral": 0.70,

    "focused": 0.90,

    "surprise": 0.60,

    "sad": 0.30,

    "angry": 0.20,

    "fear": 0.30,

    "disgust": 0.20,

    "unknown": 0.50

}



def calculate_engagement(
        behavior,
        emotion
):


    behavior = behavior.lower()

    emotion = emotion.lower()



    behavior_value = BEHAVIOR_SCORE.get(

        behavior,

        0.50

    )



    emotion_value = EMOTION_SCORE.get(

        emotion,

        0.50

    )



    # Behavior is more important than emotion

    score = (

        behavior_value * 0.75

        +

        emotion_value * 0.25

    )



    percentage = round(
        score * 100,
        2
    )



    if percentage >= 75:

        status = "engaged"


    elif percentage >= 45:

        status = "neutral"


    else:

        status = "disengaged"



    return {

        "score": percentage,

        "status": status

    }