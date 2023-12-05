import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Load and preprocess data outside of the Streamlit application
data = pd.read_csv("../data/231030_clean_table_for_analysis.csv", low_memory=False, header=0, index_col=0, na_values='n/a')

# Create the pages
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Page", ["Project", "Exploration", "Departments", "DataVizualization", "Modelling","Conclusions"])

if page == "Project":
    image = Image.open('france_road.jpg')

    st.image(image, caption='')
    st.title("Project: Road Accidents in France")
    st.markdown("""
        According to the National Road Safety Observatory (ONSR) in France, in the year 2021,
         there were 560,666 road accidents in France. This represents a 4.1% increase from the number of accidents recorded in 2020.
         Of these accidents, 100,086 occurred in the metropolitan area of France, representing 18.3% of all accidents in the country. The number of fatalities in 2021 was 3,336, while the number of serious injuries was 74,414.
        Based on that we want to predict the number of severe accidents (Hospitalized and Deaths).
        Data used in this project was from 2005 to 2018 in order to predict the values for 2019 to 2020 """)

    st.markdown("""
        **Aim:** To optimize three classification models with the best overall performances for predicting severe accidents (at least one hospitalized or killed person in an accident versus only slightly or non-injured persons), and then calibrate, evaluate and interpret all three models.
    """)
    st.markdown("""
        **Authors:** Johanna, Tiago, Tobias
    """)

elif page == "Exploration":
    st.title("Data Exploration")

    # Show slides of figures using a slider
    figure_index = st.slider('Figure Index', min_value=1, max_value=2, value=1)

    processed_data = st.cache(allow_output_mutation=True)(lambda: processed_data)

    if figure_index == 1:
        image = Image.open('exploratory_1.png')
        image_size = (1000, 400)
        st.image(image, width=image_size[0], caption='')
    elif figure_index == 2:
        image = Image.open('exploratory_2.png')
        image_size = (1000, 400)
        st.image(image, width=image_size[0], caption='')

elif page == "Departments":
    st.title("Accident Mapping")
    image = Image.open('accidents_per_department.png')
    image_size = (1000, 400)
    st.image(image, width=image_size[0], caption='')

    

elif page == "DataVizualization":
    st.title("Data Visualization")
    st.write("")    
    image = Image.open('days_week_heat.png')
    image_size = (1000, 400)
    st.image(image, width=image_size[0], caption='Accidents per Department')
    

elif page == "Modelling":
    st.title("Accident Modelling")
    # load the models here
    #maybe inlcude some checkboxs
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
   
    st.markdown("""
        **Machine Learning Application:** We can use machine learning to predict the likelihood of an accident occurring. This would be a valuable tool for road safety agencies, as it would allow them to focus their resources on areas where accidents are most likely to happen.
    """)
elif page == "Conclusion":
    st.title("Conclusion")
