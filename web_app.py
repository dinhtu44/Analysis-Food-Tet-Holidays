# %%writefile web_app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

def user_input_features(all_categorical_variables):
    num_calories = st.sidebar.slider('Calories', 0, 3257, 400)
    num_protein = st.sidebar.slider('Protein', 0, 172, 50)
    num_fat = st.sidebar.slider('Fat', 0, 207, 346)
    num_sodium = st.sidebar.slider('Sodium', 0, 5688, 2000)
    others = st.sidebar.multiselect('Other related to foods', all_categorical_variables)
    data = {'calories': num_calories,
            'protein': num_protein,
            'fat': num_fat,
            'sodium': num_sodium
            }
    for other in others:
        data[other] = 1
    input_user_df = pd.DataFrame(data, index=[0])

    for categorical_variables in all_categorical_variables:
        data[categorical_variables] = 0
    for other in others:
        data[other] = 1
    features = pd.DataFrame(data, index=[0])
    return (input_user_df, features)


def main():
    st.header("Classification of dishes for holidays and special occasions")
    st.write("""
    Use the sidebar to select input features.
    """)

    work_dir = os.getcwd()
    # thong_workspace = work_dir + '/Thong_workspace'
    # thong_model = thong_workspace + '/models'

    df = pd.read_csv('epi_new.csv')
    st.sidebar.header('User Input Features')
    all_categorical_variables = df.iloc[:,6:633].columns.tolist()
    (input_user_df, input_df) = user_input_features(all_categorical_variables)

    #Write out input selection
    st.subheader('User Input')
    st.write(input_user_df)

    #Load in model
    import joblib
    load_clf = joblib.load('Models/LR_quantile.pkl')


    # Apply model to make predictions
    if st.button("Predict"): 
        # result = prediction(sepal_length, sepal_width, petal_length, petal_width) 
        prediction = load_clf.predict(input_df)
        prediction_proba = load_clf.predict_proba(input_df)
        if prediction == 0:
            # st.success('This is a holiday food with accuracy {}'.format(prediction_proba))
            st.success('This is a holiday food')
        else:
            # st.success('This is a non-holiday food with accuracy {}'.format(prediction_proba))
            st.success('This is a non-holiday food')
    # st.subheader('Prediction')
    # st.write("""
    # This is a multi-class classification model. Options are: 
    # 1) 'NO' --> this patient was not readmitted within a year, 
    # 2) '<30' --> this patient was readmitted within 30 days, or 
    # 3) '>30' --> this patient was readmitted after 30 days. 
    # This generally corresponds to the severity of the patient's diabetes as well as the specific care, or lack thereof, during the visit.
    # """)

if __name__ == '__main__':
    main()
