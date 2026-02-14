# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 22:36:17 2026

@author: Hp
"""

import numpy as np
import pickle
import streamlit as st
import lime
from lime.lime_tabular import LimeTabularExplainer

import matplotlib.pyplot as plt



#loading the model
loaded_model, loaded_scalar, X_train_values = pickle.load(open("trained_model.sav", "rb"))

#creating a function for prediction

def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array and then reshaping it so that it appears as 2D array with 1 row and total columns
    input_data_reshaped = np.asarray(input_data).reshape(1,-1)
    
    input_data_scaled= loaded_scalar.transform(input_data_reshaped)

   
    prediction = loaded_model.predict(input_data_scaled)
    
    return prediction[0], input_data_scaled
  
#working to see lime results too.
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
class_names = ['Non Diabetic', 'Diabetic']


explainer = LimeTabularExplainer(
    training_data= X_train_values,
    feature_names= feature_names,
    class_names= class_names,
    mode='classification'
    
    )



def explain_prediction(input_scaled):
    # LIME expects the original, unscaled input for explain_instance
    # We'll need to pass it in the right format
    exp = explainer.explain_instance(
        data_row=input_scaled[0],  # use the scaled data if your explainer was trained on scaled data
        predict_fn=lambda x: loaded_model.predict_proba(loaded_scalar.transform(x)),
        num_features=8
    )
    
    # Return matplotlib figure for Streamlit
    fig = exp.as_pyplot_figure()
    return fig


def main():
    
    
    #giving a title
    
    st.title("Diabetes Prediction Web App")
    st.write("Please enter each value and press Enter before moving on to the next input field.")

    
    #working on taking the input
    #remember that the sequence of the input data should be same as that of the dataset
    #gettin the input data from the user 
    
  
    Pregnancies = st.text_input('Number of Pregenancies')
    Glucose= st.text_input('Blood Glucose level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Thickness of the skin')
    Insulin= st.text_input('Blood Insulin Level')
    BMI=st.text_input('Body Mass Index (BMI)')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function')
    Age=st.text_input('Age of the person')
    
    #code for prediction
    
    if st.button('Diabetes Test Results'):
    # Get prediction and scaled input
        pred_class, input_scaled = diabetes_prediction([
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
        BMI, DiabetesPedigreeFunction, Age
    ])

    # Show textual diagnosis
        if pred_class == 0:
            st.success('The person is not diabetic')
        else:
                st.success('The person is diabetic')
        st.title("Feature Influence on Diabetes Prediction")
        st.write("The chart below shows how much each feature affected the prediction for this person. ")
        st.write("Green bars indicate features that pushed the prediction toward being Diabetic, ")
        st.write("while red bars show features that pushed it toward Not Diabetic.")

        # Show LIME explanation graph
        original_input = np.array([
            float(Pregnancies), float(Glucose), float(BloodPressure),
            float(SkinThickness), float(Insulin), float(BMI),
            float(DiabetesPedigreeFunction), float(Age)
            ]).reshape(1, -1)

        fig = explain_prediction(original_input)
        st.pyplot(fig)

    
   
   
    
if __name__ == '__main__':

    main()
