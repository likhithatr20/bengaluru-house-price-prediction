import pandas as pd
import pickle
import streamlit as st
import os

# --- Load model safely ---
model_path = os.path.join(os.path.dirname(__file__), 'banglore_home_price_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# --- Load dataset safely ---
data_path = os.path.join(os.path.dirname(__file__), 'cleaned_data.csv')
data = pd.read_csv(data_path)

# --- Streamlit UI ---
st.header(' Bangalore House Price Prediction')

loc = st.selectbox('Choose the Location', data['location'].unique())
sqft = st.number_input('Enter the Total Square Feet', min_value=300.0, step=50.0)
beds = st.number_input('Enter the Number of Bedrooms', min_value=1, step=1)
bath = st.number_input('Enter the Number of Bathrooms', min_value=1, step=1)
balcony = st.number_input('Enter the Number of Balconies', min_value=0, step=1)

# --- Prepare input for prediction ---
input_df = pd.DataFrame([[loc, sqft, bath, balcony, beds]],
                        columns=['location', 'total_sqft', 'bath', 'balcony', 'bedrooms'])

if st.button('Predict Price'):
    try:
        output = model.predict(input_df)
        st.success(f' The predicted price of the house is {round(output[0], 2)} Lakhs')
    except Exception as e:
        st.error(f"Prediction failed: {e}")
