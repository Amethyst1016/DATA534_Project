# find numeric_cols
def numeric_cols(df):
    numeric_cols=df.select_dtypes('number').columns.tolist()
    return numeric_cols




# draw time trend plot

## lst means a list including the column name you want to choose
## for the whole years
def time_series_trend (lst,df):
    chart=alt.Chart(df).mark_line(interpolate='monotone').encode(
       x=alt.X("date",axis=None),
       y=alt.Y(alt.repeat(),type="quantitative",scale=alt.Scale(zero=False))
       ).properties(width=300,height=300).repeat(repeat=lst)
    chart.show()
    
## for the specific years    
def time_series_trend_year (lst,df,start_year, end_year):
    chart=alt.Chart(df.loc[(df["date"].dt.year <=end_year) & (df["date"].dt.year >= start_year), :]).mark_line(interpolate='monotone').encode(
       x=alt.X("date"),
       y=alt.Y(alt.repeat(),type="quantitative",scale=alt.Scale(zero=False))
       ).properties(width=300,height=300).repeat(repeat=lst)
    chart.show()
    
    
    
# draw boxplot    
def boxplot(lst,df,start_year, end_year):
    dfboxplot = df.copy()
    dfboxplot['year'] = pd.to_datetime(dfboxplot['date']).dt.year
    chart=alt.Chart(dfboxplot.loc[(dfboxplot["year"] <=end_year) & (dfboxplot["year"]>= start_year), :]).mark_boxplot().encode(
        alt.Y("year:O"),
        alt.X(alt.repeat(),type="quantitative",scale=alt.Scale(zero=False)),
        alt.Tooltip('Title:N')
        ).properties(width=500, height=500).repeat(repeat=lst)
    chart.show()
    
    

# draw corrlation--a scatterplot matrix (SPLOM)
def scatterplot_matrix(numeric_cols):
    chart=alt.Chart(df).mark_point(opacity=0.5,size=2).encode(
       alt.X(alt.repeat('column'),type="quantitative",scale=alt.Scale(zero=False)),
       alt.Y(alt.repeat('row'),type="quantitative",scale=alt.Scale(zero=False))
       ).properties(height=150,width=150).repeat(row=numeric_cols,column=numeric_cols
       ).configure_axis(titleFontSize=15,labelFontSize=5)
    chart.show()

    
      
        
# draw corrlation--correlation_plot
def correlation_plot(numeric_cols):
    corr_df = df[numeric_cols].corr("spearman").stack().reset_index(name='corr')
    chart=alt.Chart(corr_df).mark_rect().encode(
    alt.X('level_0:N',title=None),
    alt.Y('level_1:N',title=None),
    color='corr:Q'
    ).properties(width=200,height=200)
    chart.show()
    

    
