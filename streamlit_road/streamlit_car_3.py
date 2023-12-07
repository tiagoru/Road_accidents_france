import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# Load and preprocess data outside of the Streamlit application
data = pd.read_csv("../data/231030_clean_table_for_analysis.csv", low_memory=False, header=0, index_col=0, na_values='n/a')

# Create the pages
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Page", ["Project", "Exploration", "Number of victims", "DataVizualization", "Modelling","Conclusions"])

# Create a list of figures
# Create a list of figures
figures = [
    {"name": "departments", "image": "accidents_per_department.png"},
    {"name": "correlation", "image": "correlation.png"},
    {"name": "weather", "image": "exploratory_1.png"},
]

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
    st.markdown("""The data used came from the French Goverment website: https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/#community-discussions
                it consist of 4 different files that after some data manipulation(merging and cleaning) became a dataframe of 2.509.598 rows and 60 colunns, a total of 1.1 Gb of data.  """)
    st.markdown(""" Data exploration includes some examples of the data, before we choose the target variable.""" )


    # Show slides of figures using a slider
    figure_index = st.slider('Figure Index', min_value=1, max_value=4, value=1)

    processed_data = st.cache(allow_output_mutation=True)(lambda: processed_data)

    if figure_index == 1:
        st.markdown(""" Figure 1 Show the number of missing values per variables""")
        image = Image.open('Missing_values.png')
        image_size = (1000, 400)
        st.image(image, width=image_size[0], caption='percentage of missing values in the dataframe variables')
    elif figure_index == 2:
        st.markdown(""" Table 1, shows the data description of the dataframe""")
        st.table(data.describe())
      

        
        #image = Image.open('exploratory_1.png')
        #image_size = (1000, 400)
        #st.image(image, width=image_size[0], caption='')
    elif figure_index == 3:
        st.markdown(""" Figure 2 Shows the percentage types of gravity of the accidents distributed in 4 different categories in France""")
        image = Image.open('acc_types.png')
        image_size = (500, 200)
        st.image(image, width=image_size[0], caption='')

    elif figure_index == 4:
        st.markdown(""" Figure 3 Shows the correlations between the numerics variables""")
        image = Image.open('corr_raw.png')
        image_size = (500, 200)
        st.image(image, width=image_size[0], caption='')

elif page == "Number of victims":
    st.markdown(""" The figures in the menu below, shows the distribution of victims of accidents in each type per year""")
    # Create a menu to choose which figure to display
    selected_figure = st.selectbox("Choose a figure", ["Unscathed", "Light Injuries", "Hospitalized","Killed"])
    # If a figure was selected, display it
    if selected_figure:
        if selected_figure == "Unscathed":
            image = Image.open('acc_per_uns.png')
            image_size = (500, 200)
            st.image(image, width=image_size[0], caption='Unscathed')
        elif selected_figure == "Light Injuries":
            image = Image.open('acc_per_light.png')
            image_size = (500, 200)
            st.image(image, width=image_size[0], caption='Light Injuries')
        elif selected_figure == "Hospitalized":
            image = Image.open('acc_per_hosp.png')
            image_size = (500, 200)
            st.image(image, width=image_size[0], caption='Hospitalized')
        elif selected_figure == "Killed":
            image = Image.open('acc_per_kill.png')
            image_size = (500, 200)
            st.image(image, width=image_size[0], caption='Killed')

elif page == "DataVizualization":
    st.title("Data Visualization")
    st.markdown(""" Based on the distribution of the data and the previus analysis, the types of accidents were merged
                in 2 categories ( severe(hospitalized + killed) and non severe(Unscathed + Light injuries)), as target variables for our prediction model.
                this choice was made in order to balance the distribution of the target variable.
                The figures in the menu below shows the data visualization for the target variable""")
      # Create a menu to choose which figure to display
    selected_figure = st.selectbox("Choose a figure", ["Distribution by day and hour", "Accidentes per hour", "Vehicles types","Accidents per department","Correlations with target variable"])
    # If a figure was selected, display it
    if selected_figure:
        if selected_figure == "Distribution by day and hour":
            st.write("Distribution of severe accidents by weekday and hour")
            image = Image.open('days_week_heat.png')
            image_size = (500, 200)
            st.image(image, width=image_size[0], caption='')
        elif selected_figure == "Accidentes per hour":
            image = Image.open('accidents_per_hour.png')
            image_size = (500, 200)
            st.image(image, width=image_size[0], caption='')
        elif selected_figure == "Vehicles types":
            image = Image.open('distri_vehi.png')
            image_size = (500, 200)
            st.image(image, width=image_size[0], caption='')
        elif selected_figure == "Accidents per department":
            image = Image.open('accidents_per_department.png')
            image_size = (900, 400)
            st.image(image, width=image_size[0], caption='')
        elif selected_figure == "Correlations with target variable":
            image = Image.open('correlation.png')
            image_size = (500, 200)
            st.image(image, width=image_size[0], caption='')
elif page == "Modelling":
    st.title("Accident Modelling")
    st.markdown("""Our project relates to a classification machine learning problem. It is related to traffic research.
                 It predicts the severity of injuries by traffic accidents on the basis of several factors associated with the circumstances of the accident,
                 and the involved persons and vehicles. The identified target variable is grav, 
                which encodes the severity of the accidents in four classes. 
                The classes are 1: unscathed, 2: killed, 3: hospitalized, and 4: light injured. 
                In principle, this is a multi-class classification problem. 
                During the project, the classes 1+4 and 2+3 were encoded to 0: non-severe and 1: severe, stored in severe.
                 Finally, the multiclass model was reduced to a binary class model.""")

    selected_figure = st.selectbox("Choose a Model", ["Decision Tree", "Random Forest", "XGBoost"])
    # If a figure was selected, display it
    if selected_figure:
        if selected_figure == "Decision Tree":
            st.write("Decision Tree")
            st.markdown("""Decision Tree Classifier was chosen because of the high explainability and also due to the fact that the models are relatively simple to set up and train.
                         In addition to that, the performance of these models is also very good.
                         In some cases, Decision Trees tend to overfit, but this problem can be eliminated with GridSearchCV technique""")
            #need to include the result of the models and the metrics 
            
            
        elif selected_figure == "Random Forest":
            st.write("Random Forest")
            st.markdown("""Random Forest was chosen because it combines the predictions of multiple individual decision trees to make a final prediction.
                         This ensemble approach tends to provide more accurate and robust predictions compared to a single decision tree""")
           #need to include the result of the models and the metrics 
        elif selected_figure == "XGBoost":
            st.write("XGBoost")
            st.markdown("""XGBoost was chosen because it is considered as a model with high accuracy, regularization as lasso (L1) and ridge (L2) to prevent overfitting,
                         it can handle imbalanced datasets, it is flexible in data types, and it includes intrinsic feature importance which enable an easier interpretability of the results. 
                        XGBoost is considered as an advanced model.""")
            #need to include the result of the models and the metrics 

 
        #st.markdown("""**Machine Learning Application:** We  used machine learning to predict the likelihood of an accident occurring. This would be a valuable tool for road safety agencies, as it would allow them to focus their resources on areas where accidents are most likely to happen.
        #            """)
elif page == "Conclusion":
    st.title("Conclusion")
