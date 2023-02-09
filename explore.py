import pandas as pd
import altair as alt
import seaborn as sns

def get_numeric_cols(df):
    """
    Parameters:
        df: input dataframe
    
    Return: 
        list of name of numeric columns in df
    """
    if type(df) != pd.DataFrame:
        raise TypeError(f'pd.DataFrame expected but got {type(df)}')
    numeric_cols = df.select_dtypes('number').columns.tolist()
    return numeric_cols

def time_series_trend(df, lst, start_date=None, end_date=None):
    """
    Parameters:
        df: input dataframe
        lst: list of column name in df
        start_date (optional): if want to check trend of a range of time, specify start date of the range
        end_date (optional): if want to check trend of a range of time, specify end date of the range
    
    Returen:
        time trend plot (altair)
    """
    if type(df) != pd.DataFrame:
        raise TypeError(f'pd.DataFrame expected but got {type(df)}')
    if type(lst) != list:
        raise TypeError(f'list expected but got {type(lst)}')
    if start_date and end_date:
        df = df[(df['date']>=start_date) & (df['date']<=end_date)]
    chart=alt.Chart(df).mark_line(interpolate='monotone').encode(
       alt.X('date'),
       alt.Y(alt.repeat(),type='quantitative',scale=alt.Scale(zero=False))
    ).repeat(repeat=lst)
    return chart
       
def boxplot_year(df, lst, years=None):
    """
    Parameters:
        df: input dataframe
        lst: list of column name in df
        years (optional): if want to check boxplot of some years, specify the list of year
    
    Returen:
        boxplot (altair)
    """
    if type(df) != pd.DataFrame:
        raise TypeError(f'pd.DataFrame expected but got {type(df)}')
    if type(lst) != list:
        raise TypeError(f'list expected but got {type(lst)}')
    if years:
        df = df[df['date'].dt.year.isin(years)]
    chart=alt.Chart(df).mark_boxplot().encode(
        alt.Y("year(date):N"),
        alt.X(alt.repeat(),type="quantitative",scale=alt.Scale(zero=False)),
        alt.Tooltip('Title:N')
        ).repeat(repeat=lst)
    return chart

def correlation_scatter(df):
    """
    Parameters:
        df: input dataframe containing all indices
    
    Returen:
        correlation scatter plot (seaborn)
    """
    if type(df) != pd.DataFrame:
        raise TypeError(f'pd.DataFrame expected but got {type(df)}')
    return sns.pairplot(df)

def correlation_heatmap(df):
    """
    Parameters:
        df: input dataframe containing all indices
    
    Returen:
        correlation heatmap plot (altair)
    """
    if type(df) != pd.DataFrame:
        raise TypeError(f'pd.DataFrame expected but got {type(df)}')
    corr_df = df.corr("spearman").stack().reset_index(name='corr')
    corr_df['corr'] = corr_df['corr'].apply(lambda x: round(x,2))
    base=alt.Chart(corr_df).encode(
        alt.X('level_0:N',title=None),
        alt.Y('level_1:N',title=None)
    ).properties(width=250, height=250)
    chart=base.mark_rect().encode(color='corr:Q')
    text=base.mark_text(baseline='middle').encode(
        text='corr:Q',
        color=alt.condition(
            alt.datum.corr > 0,
            alt.value('white'),
            alt.value('black'))
    )
    return chart + text