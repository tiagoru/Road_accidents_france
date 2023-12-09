import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import sys

#image = Image.open("./france_road.jpg")

#st.image(image, caption='')


# load dataframe after cleaning

df=pd.read_csv("../data/231030_clean_table_for_analysis.csv", low_memory=False, header = 0, index_col=0, na_values='n/a')


# creating pages in Streamlit 

st.title("Road Accidents in France")
st.sidebar.title("Table of contents")
pages=["Project","Exploration", "DataVizualization", "Modelling", "mapping"]
page=st.sidebar.radio("Go to", pages)


# Editing the first page "Presentation of the data"
if page == pages[0]: 
    st.write("### Project")
    st.write("#### Aim: optimize three classification models with the best overall performances for predicting severe accidents (at least one hospitalized or killed person in an accident versus only slightly or non-injured persons), and then calibrate, evaluate and interpret all three models")
<<<<<<< HEAD
    st.write("##### authors: Johanna, Tiago , Tobias ")
=======
    st.write("##### authors: Johanna, Tiago, Tobias ")
>>>>>>> b4c3a1b (local adjustments)

if page == pages[1] :
    st.write("### Data description")
    st.write(df.shape)
    st.write("#### data set provide by the French goverment from 2005 to 2021")
    st.dataframe(df.describe())

    if st.checkbox("Show data types") :
        st.write(df.dtypes)




# editing visualization of the data
if page == pages[2] : 
    st.title("Data Vizualization")
    st.write("### Accidents per departments")
    
    image = Image.open('accidents_per_department.png')
    image_size = (1000, 400)
    st.image(image, width=image_size[0], caption='Accidents per Department')
    
    st.write("### Data exploration")
    
    image = Image.open('exploratory_1.png')
    image_size = (1000, 400)
    st.image(image, width=image_size[0], caption='weather and road conditions')

    st.write("### Data exploration")
    
    image = Image.open('exploratory_2.png')
    image_size = (1000, 400)
    st.image(image, width=image_size[0], caption='local and road conditions')  
  

# modeling results

if page == pages[3] : 
    st.write("### Modelling")
    st.write("### Variables Correlation")
    image = Image.open('correlation.png')
    image_size = (1000, 400)
    st.image(image, width=image_size[0], caption='correlations')

    if st.checkbox("Show XGboost model") :
        st.write("### XGboost modelling results")
        image = Image.open('Matrix_XG.png')
        #image_size = (1000, 400)
        st.image(image, width=image_size[0], caption='Confusion matrix of the XGboost model')
    if st.checkbox("Show Decision tree model") :
        st.write("### teste")
        #image = Image.open('Matrix_XG.png')
        #image_size = (1000, 400)
        #st.image(image, width=image_size[0], caption='Confusion matrix of the XGboost model')
    if st.checkbox("Show Random Forest model") :
        st.write("### teste")
        #image = Image.open('Matrix_XG.png')
        #image_size = (1000, 400)
        #st.image(image, width=image_size[0], caption='Confusion matrix of the XGboost model')
if page == pages[4] : 
    st.write("### mapping")
    def displayMe(df, grouping_keys, departement = 750):
 
 
 ## Withdrawing the global variables "’SelectedLabelsFor’+element" from the previous multiselects
 ## we could do it another better way imo, but still...

    #thismodule = sys.modules[__name__]
 
 ## for Parisian departement, 
 ## take the Selected Labels from the MultiSelect(s) widgets and filter on them.
    df_mini = df[df[‘dep’]==departement]
    for element in grouping_keys:
    st.write(globals()[‘SelectedLabelsFor’+element])
    df_mini = df_mini[  df_mini[element].isin(getattr(thismodule, ‘SelectedLabelsFor’+element)) ]## Create a plot to display the Map.
    fig, ax = plt.subplots(figsize=(14,7), nrows=1, ncols=1)
    for dataset_name, dataset in df_mini.groupby(grouping_keys):
    ax.plot(dataset.long, dataset.lat, marker=’o’, linestyle=’’, ms=6, label=dataset_name)
    ax.legend()
    the_plot_map.pyplot(plt)  ## important: to display in the Streamlit app a matplotlib.pyplot figure, 
  ## we will create later on a placeholder the_plot_map = st.pyplot(plt)
  ## This plot will be filled upon execution of the previous line 
  ## Create a plot to display other stuff...
    ig, (ax1,ax2) = plt.subplots(figsize=(14,3), nrows=1, ncols=2)
    df_mini.groupby(grouping_keys).size().unstack().plot(kind=’bar’, stacked=True, ax=ax1, title="In Paris") 
    df.groupby(grouping_keys).size().unstack().plot(kind=’bar’, stacked=True, ax=ax2, title="In France")
    the_plot_bar.pyplot(plt) ## important: to display in the Streamlit app
    
    