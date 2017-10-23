"""Functions to evaluate the tension value of a certain emotion."""
POSITIVE_HIGH = 2
POSITIVE_LOW = 1
NEGATIVE_LOW = -1
NEGATIVE_HIGH = -2
EMOTIONS_TO_TENSION = {
    "joy": POSITIVE_HIGH,
    "amusement": POSITIVE_HIGH,
    "pride": POSITIVE_HIGH,

    "pleasure": POSITIVE_LOW,
    "relief": POSITIVE_LOW,
    "interest": POSITIVE_LOW,

    "hot anger": NEGATIVE_HIGH,
    "panic fear": NEGATIVE_HIGH,
    "despair": NEGATIVE_HIGH,

    "irritation": NEGATIVE_LOW,  # called "cold anger" in the original paper
    "anxiety": NEGATIVE_LOW,
    "sadness": NEGATIVE_LOW
}


def get_tension_value(emotion):
    """Get the tension value corresponding to a given emotion.

    Parameters
    ----------
    emotion : string
        Text name of an emotion.

    Returns
    -------
    tension_value : float
        Tension value corresponding to the given emotion.

    Notes
    -----
    We use a very naive approach. Using the GEMEP emotion model, there are
    twelve possible emotions divided into four groups:
    positive valence, high arousal
    positive valence, low arousal
    negative valence, high arousal
    negative valence, low arousal
    in which the given emotions are classified.
    Then we associate a numerical value to each group on a scale from -2 to +2
    (0 corresponding to not available emotion, hence counting it as a neutral
    one).
    So, we have as input an emotion, then we associate it to the right group
    and hence give it a numerical value (tension value).

    """
    if emotion in EMOTIONS_TO_TENSION:
        tension_value = EMOTIONS_TO_TENSION[emotion]
    else:
        # if an emotion is not recognized, print a warning and invalidate
        # the result
        print("Warning: Emotion {} is not recognized.".format(emotion))
        tension_value = 0
    return tension_value
