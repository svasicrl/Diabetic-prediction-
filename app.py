import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Diabetic Risk Predictor", layout="centered")

# --- Load Model and Features ---
# The model.pkl contains the trained RandomForestClassifier model.
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# The features.pkl contains the list of feature columns the model was trained on,
# ensuring consistent input structure.
with open('features.pkl', 'rb') as file:
    model_features = pickle.load(file)

# --- Streamlit App UI ---
st.title("Diabetic Risk Predictor")
st.markdown("Enter the patient's information to predict their diabetic risk category.")

st.sidebar.header("Patient Input Features")

def get_user_input():
    # Numerical Inputs
    age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
    waist_circumference = st.sidebar.number_input('Waist Circumference (cm)', min_value=50.0, max_value=200.0, value=90.0)
    fasting_glucose = st.sidebar.number_input('Fasting Glucose (mg/dL)', min_value=50.0, max_value=300.0, value=90.0)
    fasting_triglycerides = st.sidebar.number_input('Fasting Triglycerides (mg/dL)', min_value=30.0, max_value=1000.0, value=100.0)
    tyg_index = st.sidebar.number_input('TYG Index', min_value=5.0, max_value=15.0, value=8.0, format="%.4f")

    # Score Inputs (included as they are part of model_features and no derivation logic is provided)
    age_score = st.sidebar.number_input('Age Score', min_value=0, max_value=30, value=0)
    abdominal_obesity_score = st.sidebar.number_input('Abdominal Obesity Score', min_value=0, max_value=20, value=0)
    physical_activity_score = st.sidebar.number_input('Physical Activity Score', min_value=0, max_value=10, value=0)
    family_history_score = st.sidebar.number_input('Family History Score', min_value=0, max_value=10, value=0)

    # Categorical Inputs
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    physical_activity_level = st.sidebar.selectbox('Physical Activity Level', [
        'Vigorous exercise or strenuous at work',
        'Moderate exercise at work/home',
        'No exercise and sedentary'
    ])
    family_history_diabetes = st.sidebar.selectbox('Family History of Diabetes', [
        'Either parent diabetic',
        'Two non-diabetic parents'
    ])

    user_data = {
        'Age': age,
        'Waist Circumference (cm)': waist_circumference,
        'Fasting Glucose (mg/dL)': fasting_glucose,
        'Fasting Triglycerides (mg/dL)': fasting_triglycerides,
        'TYG Index': tyg_index,
        'Age Score': age_score,
        'Abdominal Obesity Score': abdominal_obesity_score,
        'Physical Activity Score': physical_activity_score,
        'Family History Score': family_history_score,
        'Gender': gender,
        'Physical Activity Level': physical_activity_level,
        'Family History of Diabetes': family_history_diabetes
    }
    return pd.DataFrame(user_data, index=[0])

input_df_raw = get_user_input()

# --- Preprocess User Input to Match Model Features ---
def preprocess_input(input_df_raw, model_features):
    # One-hot encode categorical features similar to training data
    # Assuming 'Gender', 'Physical Activity Level', 'Family History of Diabetes' are categorical
    df_processed = pd.DataFrame(0, index=[0], columns=model_features)

    # Populate numerical features directly
    for col in ['Age', 'Waist Circumference (cm)', 'Fasting Glucose (mg/dL)',
                'Fasting Triglycerides (mg/dL)', 'TYG Index', 'Age Score',
                'Abdominal Obesity Score', 'Physical Activity Score', 'Family History Score']:
        if col in input_df_raw.columns and col in df_processed.columns:
            df_processed[col] = input_df_raw[col].values[0]

    # Handle 'Gender_Male'
    if 'Gender_Male' in df_processed.columns:
        if input_df_raw['Gender'].values[0] == 'Male':
            df_processed['Gender_Male'] = 1

    # Handle 'Physical Activity Level' one-hot encoding
    pa_level = input_df_raw['Physical Activity Level'].values[0]
    if f'Physical Activity Level_{pa_level}' in df_processed.columns:
        df_processed[f'Physical Activity Level_{pa_level}'] = 1

    # Handle 'Family History of Diabetes' one-hot encoding
    fh_diabetes = input_df_raw['Family History of Diabetes'].values[0]
    if f'Family History of Diabetes_{fh_diabetes}' in df_processed.columns:
        df_processed[f'Family History of Diabetes_{fh_diabetes}'] = 1

    return df_processed


input_for_prediction = preprocess_input(input_df_raw, model_features)

# Display User Input (optional)
st.subheader('User Input:')
st.write(input_df_raw)

# --- Make Prediction ---
st.subheader('Diabetic Risk Prediction:')

if st.button('Predict'):
    try:
        prediction = model.predict(input_for_prediction)
        st.success(f"The predicted Diabetic Risk Category is: **{prediction[0]}**")

        # Optional: Display prediction probabilities
        # prediction_proba = model.predict_proba(input_for_prediction)
        # st.write("Prediction Probabilities:")
        # proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        # st.write(proba_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("""
---
### Important Note on Model Performance

This model achieved **perfect accuracy, precision, recall, F1-score, and ROC AUC (1.00)** on the test set during development. This level of performance is highly unusual for real-world datasets and strongly suggests potential issues such as:

*   **Data Leakage**: Information from the target variable might have inadvertently influenced the features.
*   **Synthetic or Overly Simplistic Dataset**: The dataset might be too clean or simplistic, making the prediction task trivial.

While the model performs flawlessly on the provided data, caution is advised when interpreting its predictions for new, unverified data. Further investigation into the dataset's origin and potential leakage is recommended for building a robust and generalizable model.
""")
