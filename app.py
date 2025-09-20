import pandas as pd
import pickle as pk
import streamlit as st

model = pk.load(open('banglore_home_price_model.pkl', 'rb'))

st.header('Bangalore House Price Prediction')
data = pd.read_csv('cleaned_data.csv')

loc=st.selectbox('Choose the Location',data['location'].unique())
sqft = st.number_input('Enter the Total Square Feet')
beds = st.number_input('Enter the Number of Bedrooms')
bath = st.number_input('Enter the Number of Bathrooms')
balcony = st.number_input('Enter the Number of Balconies')

input = pd.DataFrame([[loc, sqft, bath, balcony, beds]], columns=['location', 'total_sqft', 'bath','balcony', 'bedrooms'])

if st.button('Predict Price'):
    output = model.predict(input)
    out_str = 'The predicted price of the house is ' + str(round(output[0], 2)) + ' Lakhs'
    st.success(out_str)