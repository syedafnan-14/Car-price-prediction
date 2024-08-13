import streamlit as st
import pandas as pd
import joblib

df = pd.read_csv("cleaned_car_details.csv")

car = df["car_model"].unique()
fuel_type = df["fuel"].unique()
seller = df["seller_type"].unique()
transmission_type = df["transmission"].unique()
deller = df["owner"].unique()

st.title("Car Price Prediction")
st.image("cars.jpeg")

left,right = st.columns(2)

car_model = left.selectbox(
    'Enter the Car Name',
    car
)

seller_type = left.selectbox(
    'Seller Type',
    seller
)

owner = left.selectbox(
    'Owner',
    deller
)

fuel = right.selectbox(
    'Fuel Type',
    fuel_type
)

transmission = right.selectbox(
    'Transmission Type',
    transmission_type
)

Age = right.slider('Age',0,30)

km_driven = st.slider("km_driven",10,1500000)

pipe = joblib.load('car_model.pkl')

new_value = pd.DataFrame(data=[[km_driven,fuel,seller_type,transmission,owner,car_model,Age]],columns=['km_driven','fuel', 'seller_type', 'transmission','owner','car_model','Age'])
st.write(new_value)
pred = pipe.predict(new_value)
st.write(pred)