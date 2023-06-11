import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and data
model = pickle.load(open('Data-WareHouse\model.pkl', 'rb'))
df = pd.read_pickle('Data-WareHouse\data.pkl')
preprocessor = pickle.load(open('Data-WareHouse\data_transformation\preprocessing.pkl', 'rb'))

st.title("Laptop Predictor")

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
    query_df = pd.DataFrame(data=[query], columns=df.columns.drop('Price'))

    # Select only the relevant columns from the query_df
    query_df = query_df[df.columns.drop('Price')]

    # Perform preprocessing on the query data
    query_transformed = preprocessor.transform(query_df)

    

    # Perform prediction using the model
    predicted_price = np.exp(model.predict(query_transformed))[0]

    st.title(f"The predicted price of this configuration is {str(int(predicted_price))} Thousand")
