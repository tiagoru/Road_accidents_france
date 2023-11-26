import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

image = Image.open('france_road.jpg')

st.image(image, caption='')


# load dataframe after cleaning

df=pd.read_csv("231030_clean_table_for_analysis.csv", low_memory=False, header = 0, index_col=0, na_values='n/a')


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

    if st.checkbox("Show NaN") :
        st.dataframe(df.isna().sum())




# editing visualization of the data
if page == pages[2] : 
    st.write("### DataVizualization")
    fig = plt.figure(figsize=(18, 8))
    # Plot for Severe Cases
    
    sns.countplot(x='an', data=df[df['grav'] == 4], palette='viridis')
    plt.xlabel('Year')
    plt.ylabel('Number of fatal Cases')
    plt.title('Number of fatal Cases (grav=4) per Year')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # department accidents
    department_mapping = {
    'Ain': 1, 'Aisne': 2, 'Allier': 3, 'Alpes-de-Haute-Provence': 4, 'Hautes-Alpes': 5,
    'Alpes-Maritimes': 6, 'Ardèche': 7, 'Ardennes': 8, 'Ariège': 9, 'Aube': 10,
    'Aude': 11, 'Aveyron': 12, 'Bouches-du-Rhône': 13, 'Calvados': 14, 'Cantal': 15,
    'Charente': 16, 'Charente-Maritime': 17, 'Cher': 18, 'Corrèze': 19,
    '2A Cor(Ajaccio)(Bastia)': 20, 'Côte dDOr': 21,
    'Côtes dArmor': 22, 'Creuse': 23, 'Dordogne': 24, 'Doubs': 25, 'Drôme': 26,
    'Eure': 27, 'Eure-et-Loir': 28, 'Finistère': 29, 'Gard': 30, 'Haute-Garonne': 31,
    'Gers': 32, 'Gironde': 33, 'Hérault': 34, 'Ille-et-Vilaine': 35, 'Indre': 36,
    'Indre-et-Loire': 37, 'Isère': 38, 'Jura': 39, ' Landes': 40, 'Loir-et-Cher': 41,
    'Loire': 42, 'Haute-Loire': 43, 'Loire-Atlantique': 44, 'Loiret': 45, 'Lot': 46,
    'Lot-et-Garonne': 47, 'Lozère': 48, 'Maine-et-Loire': 49, 'Manche': 50, 'Marne': 51,
    'Haute-Marne': 52, 'Mayenne': 53, 'Moselle': 54, 'Meuse': 55, 'Morbihan': 56,
    'Meurthe-et-Moselle': 57, 'Nièvre': 58, 'Nord': 59, 'Oise': 60, 'Orne': 61,
    'Pas-de-Calais': 62, 'Puy-de-Dôme': 63, 'Pyrénées-Atlantiques': 64,
    'Hautes-Pyrénées': 65, 'Pyrénées Orientales': 66, 'Bas-Rhin': 67, 'Haut-Rhin': 68,
    'Rhône': 69, 'Haute-Saône': 70, 'Saône-et-Loire': 71, 'Sarthe': 72, 'Savoie': 73,
    'Haute-Savoie': 74, 'Paris': 75, 'Seine-Maritime': 76, 'Seine-et-Marne': 77,
    'Yvelines': 78, 'Deux-Sèvres': 79, 'Somme': 80, 'Tarn': 81, 'Tarn-et-Garonne': 82,
    'Var': 83, 'Vaucluse': 84, 'Vendée': 85, 'Vienne': 86, 'Haute-Vienne': 87,
    'Vosges': 88, 'Yonne': 89, 'Territoire de Belfort': 90, 'Essonne': 91,
    'Hauts-de-Seine': 92, 'Seine-Saint-Denis': 93, 'Val-de-Marne': 94, 'Val-d\'Oise': 95}
    # Reverse the department_mapping dictionary to map department numbers to names
    department_mapping_reverse = {v: k for k, v in department_mapping.items()}


    event_counts = df['dep'].value_counts()
    acci_departments = event_counts.sort_values(ascending=False)

    # Map department codes to department names for the x-axis labels
    mapped_labels = [department_mapping_reverse.get(dep_code, dep_code) for dep_code in acci_departments.index]

    # Plot the severity distribution
    fig = plt.figure(figsize=(18, 8))
    ax = acci_departments.plot(kind='bar')
    ax.set_xlabel('French Department')
    ax.set_ylabel('Number of  Cases')
    ax.set_title(' Cases by Department')
    ax.set_xticks(range(len(mapped_labels)))
    ax.set_xticklabels(mapped_labels, rotation=90)

    st.pyplot(fig)




    #plot vehicules type
    fig =plt.figure(figsize=(15, 8))
   
    fatal_grav_3_4 = df[df['grav'].isin([3, 4])]

    # Calculate event percentages
    event_counts = fatal_grav_3_4['catv'].value_counts()
    event_percentages = (event_counts / event_counts.sum()) * 100

    # Filter categories with percentage above 1% because we have more than 50 categories here
    filtered_event_percentages = event_percentages[event_percentages > 1]

    # Plot the distribution of vehicle categories for fatal cases with grav 3 or 4
    filtered_event_percentages.plot.bar()

    labels =['LV only', 'moto 125', 'moped','bicycle', 'scooter < 50', 'LCV','scooter > 50','moto >50' ,'moto 2006','scooter > 125']
    plt.title('Vehicles Categories Distribution in Fatal Cases (grav=3 or grav=4) - Above 1%')
    plt.xticks(range(len(filtered_event_percentages)),labels, rotation=0)
    plt.xlabel('Vehicle Categories')
    plt.ylabel('Count %')
    st.pyplot(fig)

# modeling results

if page == pages[3] : 
    st.write("### modelling")
    st.write("### Under contruction")




