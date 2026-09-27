import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.sav')
scaler = joblib.load('scaler.sav')

st.set_page_config(page_title="Bank Customer Churn Prediction")

st.title("Bank Customer Churn Prediction App")
st.write("Enter customer details to predict if they will churn.")

# Sidebar for user input
st.sidebar.header("Customer Input Features")

def user_input_features():
    credit_score = st.sidebar.slider('Credit Score', 350, 850, 650)
    country_option = st.sidebar.selectbox('Country', ('France', 'Germany', 'Spain'))
    gender_option = st.sidebar.selectbox('Gender', ('Female', 'Male'))
    age = st.sidebar.slider('Age', 18, 92, 35)
    tenure = st.sidebar.slider('Tenure (years)', 0, 10, 5)
    balance = st.sidebar.number_input('Balance', 0.00, 250000.00, 50000.00, step=100.00)
    products_number = st.sidebar.slider('Number of Products', 1, 4, 1)
    credit_card = st.sidebar.selectbox('Has Credit Card?', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    active_member = st.sidebar.selectbox('Is Active Member?', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    estimated_salary = st.sidebar.number_input('Estimated Salary', 0.00, 200000.00, 100000.00, step=100.00)

    # Map categorical features back to numerical encoding used in training
    country_mapping = {'France': 0, 'Germany': 1, 'Spain': 2}
    gender_mapping = {'Female': 0, 'Male': 1}

    country = country_mapping[country_option]
    gender = gender_mapping[gender_option]

    data = {
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

# Ensure the input_df columns are in the same order as X (training data columns)
# You might want to save X.columns during training and load it here for robustness
# For this example, we'll assume the order is maintained based on the user_input_features function.

# Scale the input features
scaled_input = scaler.transform(input_df)

# Predict button
if st.button('Predict Churn'):
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.subheader('Prediction')
    churn_status = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f"The customer is predicted to churn: **{churn_status}**")

    st.subheader('Prediction Probability')
    st.write(f"Probability of NOT churning (0): {prediction_proba[0][0]:.4f}")
    st.write(f"Probability of churning (1): {prediction_proba[0][1]:.4f}")

    if prediction[0] == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is likely to stay.")
