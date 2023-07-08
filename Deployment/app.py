import streamlit as st
import pickle
import numpy as np
import pandas as pd
from Deployment.design import remove

# Import the model
model = pickle.load(open('notebook/model/preprocess_model.pkl', 'rb'))
df = pickle.load(open('notebook/model/df.pkl', 'rb'))

st.set_page_config(page_title="Laptop Price Prediction ", page_icon=":💻:", initial_sidebar_state="expanded")

remove()

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                                '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu_brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())

# OS
os = st.selectbox('OS', df['Os'].unique())

if st.button('Predict Price'):
    # Preprocess the input data
    touchscreen_value = 1 if touchscreen == 'Yes' else 0
    ips_value = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = np.array([company, laptop_type, ram, weight, touchscreen_value, ips_value, ppi, cpu, hdd, ssd, gpu, os])
    columns = df.drop('Price', axis=1).columns  # Exclude the 'Price' column
    query_df = pd.DataFrame(data=query.reshape(1, -1), columns=columns)

    # Perform prediction using the preprocess_model Pipeline
    predicted_price = np.exp(model.predict(query_df))[0]

    st.title(f"The predicted price of this configuration is  {str(int(predicted_price))} Thousand")
