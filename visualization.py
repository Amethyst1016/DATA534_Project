import altair as alt
from model import *
import numpy as np
import math

def compare(models, x, y):
    """
    Parameters:
        models: List
            List of models want to compare true and prediction values
        x: str
            Name of variable to map on x axis
        y: str
            Name of variable to map on y axis
    Return:
        Line plot comparing models' performance (altair)
    """
    models_df = []
    for i in models:
        models_df.append(i.get_combined_df())
    df = pd.concat(models_df).drop_duplicates()

    if len(models) == 1:
        if models[0].simple:
            base = alt.Chart(df, title = models[0].equation).encode(
                        alt.X(x),
                        alt.Y(y),
                        color = 'type')
    else:
        base = alt.Chart(df).encode(
            alt.X(x),
            alt.Y(y),
            color = 'type')
    chart = base.mark_line(
        interpolate = 'monotone', opacity=0.6)
    return chart