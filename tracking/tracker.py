from deep_sort_realtime.deepsort_tracker import DeepSort

from config.setting import (
    MAX_AGE,
    N_INIT,
    MAX_COSINE_DISTANCE
)



def create_tracker():


    tracker = DeepSort(

        max_age=MAX_AGE,

        n_init=N_INIT,

        max_cosine_distance=
        MAX_COSINE_DISTANCE,

        embedder="mobilenet"
    )


    return tracker