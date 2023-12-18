import streamlit as st
import pandas as pd
from PIL import Image


# Create the pages
st.sidebar.title("Navigation Menu")
page = st.sidebar.selectbox("Page", ["Project", "Data Exploration", "Number of victims", "DataVizualization", "Modelling","Models","Conclusions"])

# Create a list of figures
# Create a list of figures
figures = [
    {"name": "departments", "image": "accidents_per_department.png"},
    {"name": "correlation", "image": "correlation.png"},
    {"name": "weather", "image": "exploratory_1.png"},
]

if page == "Project":
    image = Image.open('france_road.png')
    st.image(image, caption='', use_column_width = 'auto')
    st.markdown("""
    _Source: Generated on https://myedit.online_
    """)

    st.title("Project: Car Accidents in France")
    st.markdown("""
        According to the National Road Safety Observatory (ONSR) in France, in the year 2021,
        3,219 people were killed on French roads, in mainland France and overseas. 
        This figure, 8% lower than in 2019, is historically low (excluding the atypical year 2020).
        The upturn in travel in 2021 was tempered by a period of confinement in April, curfews throughout the first half of the year,
        and several periods when working from home was strongly recommended.

        Data used in this project was from 2005 to 2018 in order to predict the values for 2019 to 2020* """)
    link1 = "https://datascientest.com/"
    st.markdown(f' This is the final project for the DATA SCIENCE course at "[datascientest]({link1})."')
    st.markdown("""
        **Aim:** To optimize three classification models with the best overall performances for predicting severe accidents (at least one hospitalized or killed person in an accident versus only slightly or non-injured persons), and then calibrate, evaluate and interpret all three models.
    """)
    st.markdown("""
        **Authors:**
        
        Johanna Starkl
        
        Tiago Russomanno
        
        Tobias Schulze
    """)

elif page == "Data Exploration":
    st.title("Data Exploration")
    link2 = "https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/#community-discussions"
    st.markdown(f'The data was downloaded from the French Government website: "[link]({link2})"')
    st.markdown("""It consists of 4 different files that contain information about accidents in France, related 
                to place, time,conditions, type of vehicle, weather, type of accidents and so on.
                After data manipulation (merging and cleaning), the datframe was reduceed to 2,421,684 rows and 60 
                columns, totaling 1.1 GB of data.""")
    st.markdown("""Data exploration includes some examples of the data before we choose the target variable.""")    

    # Create a menu to choose which figure to display
    selected_figure = st.selectbox("Choose a figure", ["Missing Values", "Data description", "Type of Accidents","Weather","Variables correlations"])

  
    if selected_figure:
        if selected_figure == "Missing Values":
            st.markdown(""" Figure 1 Show the number of missing values per variables""")
            image = Image.open('missing_values.png')
            #image_size = (1000, 400)
            st.image(image, caption='percentage of missing values in the dataframe variables', use_column_width = 'auto')#width=image_size[0],
        
        elif selected_figure == "Data description":
            st.markdown(""" Table 1, shows the data description of the dataframe (subsample of ~1% of the dataframe)""")
            # Load and preprocess data outside of the Streamlit application
            n=100
            data = pd.read_csv("../data/231030_clean_table_for_analysis.csv",
                               low_memory=False,
                               header=0,
                               index_col=0,
                               na_values='n/a',
                               skiprows=lambda i: i % n != 0)
            st.dataframe(data.describe())
      
        elif selected_figure == "Type of Accidents":
            st.markdown(""" Figure 2 Shows the percentage types of the accidents gravity distributed in 4 different categories in France""")
            image = Image.open('acc_types.png')
            #image_size = (600, 300)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='')

        elif selected_figure == "Weather":
            st.markdown(""" Figure 3 Shows the weather conditions""")
            image = Image.open('weather.png')
            #image_size = (1000, 500)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='')

        elif selected_figure == "Variables correlations":
            st.markdown(""" Figure 4 Shows the correlations between the numerics variables""")
            image = Image.open('corr_raw.png')
            #image_size = (1000, 500)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='')
        
elif page == "Number of victims":
    st.markdown(""" The figures in the menu below, shows the distribution of victims of accidents in each type per year""")
    # Create a menu to choose which figure to display
    selected_figure = st.selectbox("Choose a figure", ["Unscathed", "Light Injuries", "Hospitalized","Killed"])
    # If a figure was selected, display it
    if selected_figure:
        if selected_figure == "Unscathed":
            image = Image.open('acc_per_uns.png')
            # image_size = (500, 200)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='Unscathed')
        elif selected_figure == "Light Injuries":
            image = Image.open('acc_per_light.png')
            #image_size = (500, 200)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='Light Injuries')
        elif selected_figure == "Hospitalized":
            image = Image.open('acc_per_hosp.png')
            #image_size = (500, 200)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='Hospitalized')
        elif selected_figure == "Killed":
            image = Image.open('acc_per_kill.png')
            #image_size = (500, 200)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='Killed')

elif page == "DataVizualization":
    st.title("Data Visualization")
    st.markdown("""
    The pre-analysis of the data unravelled bad modelling performances with the multi-class target `grav`
    (data not shown).
    
    The variable `grav` was merged to a binary variable `severe` based on the distribution of the data:
         
    - 0: severe (hospitalized + killed)
    - 1: non severe (unscathed + light injuries)
                
    This choice was made in order to balance the distribution of the target variable.
    
    The figures in the menu below shows the data visualization for the target variable.""")
      # Create a menu to choose which figure to display
    selected_figure = st.selectbox("Choose a figure", ["Accidentes per hour", "Distribution by day and hour", "Vehicles types","Accidents per department","Correlations with target variable"])
    # If a figure was selected, display it
    if selected_figure:
        if selected_figure == "Accidentes per hour":
            st.write("Distribution of severe accidents by weekday and hour")
            image = Image.open('accidents_per_hour.png')
            # image_size = (500, 200)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='')
        elif selected_figure == "Distribution by day and hour":
            image = Image.open('days_week_heat.png')
            # image_size = (1000, 500)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='')
        elif selected_figure == "Vehicles types":
            image = Image.open('distri_vehi.png')
            # image_size = (800, 400)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='')
        elif selected_figure == "Accidents per department":
            image = Image.open('acc_dep_top20.png')
            # image_size = (1000, 500)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='')
            #image = Image.open('map_france.png')
            # image_size = (1000, 500)
            #st.image(image, use_column_width = 'auto')#width=image_size[0], caption='')
            def main():
                st.title("Accidents per department")

                # Path to the SVG file
                svg_path = "france_map8.svg"
    

                # Read the content of the SVG file
                with open(svg_path, "r") as f:
                     svg_content = f.read()

                # Display SVG content using HTML with adjusted size
                st.markdown(f'<div style="width: 1000px; height: 800px;">{svg_content}</div>', unsafe_allow_html=True)

            if __name__ == "__main__":
                    main()
        elif selected_figure == "Correlations with target variable":
            image = Image.open('correlation.png')
            # image_size = (1000, 400)
            st.image(image, use_column_width = 'auto')#width=image_size[0], caption='')
elif page == "Modelling":
    st.title("Modelling project on a glance")
    st.markdown("""
    #### Goal
    The project relates to a to a **classification machine learning problem** in *traffic research*.
    
    It predicts the **severity of injuries** by traffic accidents on the basis of several 
    factors associated with the circumstances of the accident,
    and the involved persons and vehicles.

    #### Models
  
   - Decision Tree (Johanna)
   - Random Forest (Tiago)
   - XGBoost (Tobias)
   
    #### Target variable engineering
    
    The identified *target variable* is `grav`, which encodes the severity of the accidents in four classes:

    - 1: unscathed
    - 2: killed
    - 3: hospitalized
    - 4: light injured
        
    In principle, this is a **multi-class classification** problem.
    However, the performance of the multi-class classification modells were not sufficient. 
    
    Therefore the `grav` variable was engineered to a **binary** variable `severe` by merging classes:
    
    - 0: non-severe (classes 1 + 4)
    - 1: severe (classes 2 + 3)
    
    The multi-class classification problem is now reduced to a **binary classification** problem.
       
    #### Performance metrics and visualization
    - Accuracy
    - Precision
    - F1 score
    - Recall
    - Confusion matrix
    
    #### Feature importance and dimension reduction
    - SHAP modelling
    - principal component analysis (not shown)

 
    """)
elif page == "Models":
    st.title("Models")
    selected_figure = st.selectbox("Choose a Model to show the results", ["Model comparison of target: grav", "Decision Tree", "Random Forest", "XGBoost", "Model comparison of target: severe"])
    # If a figure was selected, display it
    if selected_figure:
        if selected_figure == "Model comparison of target: grav":
            st.markdown("""
            #### Model comparison of target variable: grav
            - First modelling was based on target variable **`grav`**
            - Variable encodes in **4 classes**, which required **multi-class modelling**
            - All three models were run.
            
            #### Findings
            Table 1 lists the performance metrics for the XGBoost and Random Forest models.

            Table 1: Performance metrics of the XGBoost and RF models using the `grav` variable as a target (best values are high-lighted)
            """)

            data = pd.read_csv('./modelling_results_1.csv', na_values="NA")
            st.dataframe(data.style.highlight_max(axis=0), hide_index = True, use_container_width = True)

            st.markdown("""
            - **Performance metrics** were **not satisfactory** and non-conforming
            - Random forest performed best in all metrics accept *Precision*
            - Decision Tree did not converge at all (stopped after 6 hours of run)
            - Class **`unscathed`** is highly **biasing** the results (Figure 1)
            """)

            image = Image.open('./xgboost_confusion_matrix_2.png')
            st.image(image, use_column_width='auto')

            st.markdown("""
            Figure 1: Confusion matrix of the XGBoost model using the `grav` variable as a target
            
            ##### Conclusions
            - Target **`grav`** is highly **imbalanced** impacting the **multi-class modelling** results
            - Requirement of further **feature engineering** to reduce complexity and gaining balance
            - Encoding of `grav` to a binary variable `severe` enables a **binary classification problem**
            """)
        
        
        
        elif selected_figure == "Decision Tree":
            st.markdown("""
            ### Decision Tree
            
            
            #### Introduction
            
            Decision Tree Classifier was chosen because of the high explainability and also due to the
            fact that the models are relatively simple to set up and to train.
            In addition to that, the performance of these models is also very good.
            In some cases, Decision Trees tend to overfit, but this problem can be eliminated with GridSearchCV technique.
            
             #### Hyperparameter Tuning with GridSearchCV 
             
             Since the performance of a model significantly depends on the value of hyperparameters it is necessary to determine the optimal values for a given model, in this case for the Decision Tree Model. And since there is no way to know in advance the best values for hyperparameters, it is also necessary to try all possible values to reach the best parameters. The solution for this is called GridSearchCV. It is the process of performing hyperparameter tuning in order to find the best parameters.
             
             The hyperparameters to be tuned are:
             - `criterion`
             - `max_depth` 
             - `min_samples_leaf`
             - `min_samples_split` 
             - `ccp_alpha`
             
             
             #### Optimal parameter for `max_depth`
             
             Since the result of the best parameter for max_depth is always the maximum of the predefined range, i.e. 10 for `max_depth`: [2, 4, 6, 8, 10] or 8 for `max_depth`: [2, 4, 6, 8], the aim was to find out, at which point the accuracy doesn't improve further. Therefore the evaluation metric AUC (Area Under Curve) was chosen. AUC provides an aggregate measure of performance across all possible classification thresholds.""")
             
             
            image = Image.open('./AUC_DT.png')
            st.image(image, use_column_width = 'auto')
            
            st.markdown("""
            
            
            
            #### Results and Interpretation

            The performance of the optimized Decision Tree model was assessed using key metrics such as the **classification report**, **accuracy score**, and a **confusion matrix**.
            These metrics provide insights into the model's precision, recall, F1-score, accuracy, and how well it performed on different classes. 
            The confusion matrix offers a detailed view of the model's true positive, true negative, false positive, and false negative predictions.

            
            ##### Classification report and accuracy score
            
            """)
            image = Image.open('./model_report_DT.png')
            st.image(image, use_column_width = 'auto')

            
            

            st.markdown("""
            
            
           The Decision Tree model demonstrates a good overall performance with an accuracy of 75%.
             
   
            ##### Confusion matrix
            
            """)
            image = Image.open('./confusion_matrix_DT.png')
            st.image(image, use_column_width = 'auto')

            st.markdown("""
            
            The diagonal elements of the matrix show that **67%** of the **"non-severe"** class and **82%** of the **"severe"** class are predicted correctly. In the first stage, the model is pretty good. In a second stage, consideration could be given to how the model should be adapted in order to deliver even better results. 
            
          
            """)
          
            
        elif selected_figure == "Random Forest":
            st.markdown("""
            ### Random Forest
            #### Introduction

            Random Forest is a powerful ensemble learning algorithm used for both classification and regression tasks.
            It operates by constructing a multitude of decision trees during training and outputs the mode (classification) or mean prediction (regression) of the individual trees.

            #### Characteristics of Random Forest

            - High accuracy
            - Robust to overfitting
            - Handles imbalanced datasets well
            - Accommodates various data types
            - Provides built-in feature importance

            #### Model Hyperparameter Optimization

            The Random Forest model underwent hyperparameter optimization using techniques like grid search and random search. 
            The goal was to find the combination of hyperparameters that maximizes the model's performance.

            #### Tuning Space

            The Random Forest hyperparameter tuning involved exploring parameters such as the number of trees (`n_estimators`), the maximum depth of the trees (`max_depth`),
            and the minimum number of samples required to split an internal node (`min_samples_split`).

            #### Results and Interpretation

            The performance of the optimized Random Forest model was assessed using key metrics such as the **classification report**, **accuracy score**, and a **confusion matrix**.
            These metrics provide insights into the model's precision, recall, F1-score, accuracy, and how well it performed on different classes. 
            The confusion matrix offers a detailed view of the model's true positive, true negative, false positive, and false negative predictions.

            ##### Classification report and accuracy score
            """)
            image = Image.open('./model_reportRF.png')
            st.image(image, use_column_width = 'auto')


            st.markdown("""
            
            - In summary, the Random Forest model demonstrates good overall performance with a decent accuracy of 77.7%.
            It exhibits balanced precision and recall for both classes, suggesting that it is effective in making accurate predictions across different outcomes
            
    
            ##### Confusion matrix
            """)
            image = Image.open('./confusionRF.png')
            st.image(image, use_column_width = 'auto')

            st.markdown("""
            
            - The confusion matrix shows a good rate of true positive and true negative values.
            - the model is better at identifying instances of the "severe" class,
            but there is room for improvement in reducing false positives for the "severe-fatal" class.

            ##### Cross-validation 
            """)
            image = Image.open('./cross_RF.png')
            st.image(image, use_column_width = 'auto')

            st.markdown("""
            
            - The F1 scores for each fold indicate the model's performance on different subsets of the training data.
            - The values range from approximately 0.7847 to 0.7931, showing consistency in the model's ability to balance precision and recall across folds.
            - Stable and good performance based on the F1 macro scores obtained through cross-validations""")


           
        elif selected_figure == "XGBoost":
            st.markdown("""
            ### XGBoost
            #### Introduction
            
            E**X**treme **G**radient **Boost**ing (XGBoost) is an **advanced** optimized distributed gradient
            boosting model used for supervising learning problems.
            
            #### Characteristics of XGBoost
            
            - high accuracy
            - regularization as lasso (L1) and ridge (L2) to prevent overfitting
            - enables handling of imbalanced datasets
            - flexible on data types
            - build-in feature importance
            
            #### Model hyperparameter optimization
            
            The XGBoost model was optimized using the **Tree-based Parzen Estimator** (TPE) combined with a **Bayesian
            Sequential Model Based Optimisation** (SMBO) and **random search on the parameter grid**.
            
            This optimization combines random search on the parameter grid with a determination of future points based
            on prior modelling results.
            
            #### Tuning space
            
            The **booster** was `gbtree` and the **evaluation metric** was `logloss` (in case of the binary model).
            
            The tuning space included several parameters such as `eta`, `reg_alpha`, `reg_lambda` and many others.
            
            #### Results and interpretation
            The result of the optimized model were interpreted using the **classification report**, **accuracy score**,
            and a **confusion matrix**.
            
            ##### Classification report and accuracy score
            """)
            image = Image.open('./xgboost_class_report.png')
            st.image(image, use_column_width = 'auto')

            st.markdown("""
            
            - An **accuracy score of 0.79** is good for a model with imbalanced, partially sparse and possible inaccurate data.
            - The **precision**, **recall** and **f1-score** of the model are quite **balanced**.
                
            ##### Confusion matrix
            """)
            image = Image.open('./xgboost_confusion_matrix.png')
            st.image(image, use_column_width = 'auto')

            st.markdown("""
            Figure 1: Confusion matrix of the XGBoost model using the `severe` variable as a target
            
            - The confusion matrix shows a **good rate** of **true positive** and **true negative** values.
            - The rates of **false positives** and **false negatives** reflect the overall **imbalance** of the dataset.
            
            ##### Conclusions
            - Reduction to a **binary classification** problem was **successfully** (accuracy of the the multi-class model was 0.51).
            - Further **predicate feature engineering** could improve the model (for instance by focussing on a curtain
            type of accident).
            
            #### Model interpretation using SHAP
            SHAP (**SH**apley **A**dditive ex**P**lanations) is an approach based on the game theory.
            It helps to interpret the outcomes of a given machine learning model by a fair allocation
            of the importance of a feature for the model outcome.
            
            The figure 2 shows the ranking of features largest influence on the modelling aggregated by SHAP values.

            """)

            image = Image.open('./xgboost_shap_values.png')
            st.image(image, use_column_width='auto')

            st.markdown("""
            Figure 2: Ranking of the most important features based on SHAP values
            
            Overall,
            - 9 features ranked high: `catv`, `obsm`, `catu`, `sexe`, `choc`, `manv`, `place`,
            - 2 variables are biased: `num_veh` and `obs` (meaning changed over time), and
            - 29 other features had only a low impact on the model.
            
            The variables encode:

            - `catv`: vehicle category
            - `obsm`: mobile object involved
            - `catu`: category of person (driver, passenger, pedestrian)
            - `sexe`: (binary) gender of the user
            - `obs`: fixed object involved (to be used with caution)
            - `choc`: chock point on the vehicle
            - `num_veh`: number of involved vehicle (to be used with caution)
            - `manv`: action prior to the accident (e.g. driving direction)
            - `place`: place of passenger in vehicle
            
            ##### Conclusion
            The highlighted features are majorly plausible, for example, a person in a van might be less affected by
            an accident that a pedestrian or cyclist.


            """)

        elif selected_figure == "Model comparison of target: severe":
            st.markdown("""
            #### Model comparison of target variable: severe
            - **Improved** modelling was based on target variable **`severe`**
            - Variable encodes in two classes enabling **binary classification** approaches
            - All three models were run.
                        
            #### Findings
         
            The comparison of thhe modelling result are listed in Figure 1.
            
            """)
            image = Image.open('./models_metrics_comparison.png')
            st.image(image, use_column_width='auto')

            st.markdown("""
            Figure 1: Performance metrics of the XGBoost, Random Forest and Decision Tree models using the `severe` variable 
            as a target
            
            - The performance of **all** three models **increased**.
            - The models ranked with a **good accuracy** in the order **XGBoost**, **Random Forest**, **Decision Tree**
            
            #### Interpretability
            Table 1 lists the common important features of all three models
            
            Table 1: Important common features supporting the models (highlighted: features not important in all models)
           """)
            data = pd.read_csv('./feature_importance.csv', na_values="NA")

            def highlight_no(s):
                return ['background-color: yellow' if v == 'no' else '' for v in s]

            data_styled = data.style.apply(highlight_no)

            st.dataframe(data_styled, hide_index=True, use_container_width=True)
            st.markdown("""
            
            - The feature importance and SHAP modelling unravelled **7 common** and **2 not in common** features
            - The **most important** feature in all models was **`catv`** (vehicle category)
      
  
            """)

        #st.markdown("""**Machine Learning Application:** We  used machine learning to predict the likelihood of an accident occurring. This would be a valuable tool for road safety agencies, as it would allow them to focus their resources on areas where accidents are most likely to happen.
        #            """)
elif page == "Conclusions":
    st.title("Final Conclusions")
    st.markdown("""
    - **All models converged** and provide a classification of the accident severity with a **good performance**  
    - **Feature engineering** reduced complexity and helped to improve the modelling
    - **XGBoost** performed best with **83%** of true positive predictions of the **severe** class
    - **Feature importance** unravelled **plausible** variables supporting the models
    - Further engineering of **predictive variables** may help to simplify the models
    
        """)

    image = Image.open('future_road.png')

    st.image(image, use_column_width='auto')

    st.markdown(
        #
        '<div style="font-size:20pt;">'
        '<div style="text-align: center;">'
        '<div style="color: red;">'
        'Thank you for your attention!'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True)

    st.markdown(
        #
        '<div style="font-size:20pt;">'
        '<div style="text-align: center;">'
        '<div style="color: red;">'
        'Seasonal Greetings and Happy New Year!'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True)

    st.markdown("""
    #
    """)

    image = Image.open('good_bye.png')

    st.image(image, use_column_width='auto')

    st.markdown("""
    _Source: Generated on https://myedit.online_
 """
   )
