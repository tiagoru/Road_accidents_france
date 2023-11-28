import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

image = Image.open('france_road.jpg')

st.image(image, caption='')


# load dataframe after cleaning

df=pd.read_csv("../data/231030_clean_table_for_analysis.csv", low_memory=False, header = 0, index_col=0, na_values='n/a')


# creating pages in Streamlit 

st.title("Road Accidents in France")
st.sidebar.title("Table of contents")
pages=["Project","Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)


# Editing the first page "Presentation of the data"
if page == pages[0]: 
    st.write("### Project")
    st.write("#### Aim: optimize three classification models with the best overall performances for predicting severe accidents (at least one hospitalized or killed person in an accident versus only slightly or non-injured persons), and then calibrate, evaluate and interpret all three models")
    st.write("##### authors: Johanna, Tiago Tobias ")

if page == pages[1] :
    st.write("### Data description")
    st.write(df.shape)
    st.write("#### data set provide by the French goverment from 2005 to 2021")
    st.dataframe(df.describe())

    if st.checkbox("Show data types") :
        st.write(df.dtypes)




# editing visualization of the data
if page == pages[2] : 
    st.write("### DataVizualization")
    image = Image.open('accidents per department.png')
    image_size = (1000, 400)
    st.image(image, width=image_size[0], caption='Accidents per Department')
      

# modeling results

if page == pages[3] : 
    st.write("### modelling")
    st.write("### variables correlation")
    image = Image.open('correlation.png')
    image_size = (1000, 400)
    st.image(image, width=image_size[0], caption='correlations')

    if st.checkbox("Show XGboost model") :
        st.write("### XGboost modelling results")
        image = Image.open('Matrix_XG.png')
        #image_size = (1000, 400)
        st.image(image, width=image_size[0], caption='Confusion matrix of the XGboost model')
