import altair as alt
from model import *

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
    # error handling
    if type(models) != list:
        raise TypeError(f'list is expected but got {type(models)}')
    if type(x) != str:
        raise TypeError(f'str is expected but got {type(x)}')
    if type(y) != str:
        raise TypeError(f'str is expected but got {type(y)}')
        
    models_df = []
    for i in models:
        models_df.append(i.get_combined_df())
    df = pd.concat(models_df).drop_duplicates()
    
    # error handling
    if y not in df:
        raise ValueError(f'\'{y}\'(y) is not in the given model(s)')
    if x not in df:
        raise ValueError(f'\'{x}\'(x) is not in the given model(s)')

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