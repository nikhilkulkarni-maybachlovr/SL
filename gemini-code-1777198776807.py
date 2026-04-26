import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Loading Data - FIXED
@st.cache_data
def load_data():
    # Attempt to load the Titanic dataset
    try:
        # Assuming the file is named 'train.csv' in your directory
        df = pd.read_csv('train.csv')
        return df
    except FileNotFoundError:
        # Fallback if file isn't found locally
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        return pd.read_csv(url)

df = load_data()

# App Title
st.title("🚢 Machine Learning from Disaster")

# Sidebar for navigation
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "Data Exploration", "Model Training"])

if app_mode == "Home":
    st.subheader("Welcome to the Analysis")
    st.write("This app explores the Titanic dataset and predicts survival.")

elif app_mode == "Data Exploration":
    st.subheader("📊 Exploratory Data Analysis")
    
    # This line was causing the error because df was None
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Dataset Preview")
            st.dataframe(df.head()) # Now works as df is a DataFrame
        
        with col2:
            st.write("### Data Statistics")
            st.write(df.describe())
            
        st.write("### Survival by Gender")
        fig = px.histogram(df, x="Sex", color="Survived", barmode="group")
        st.plotly_chart(fig)
    else:
        st.error("Data failed to load. Please check your data source.")

elif app_mode == "Model Training":
    st.subheader("🤖 Model Training")
    # Logic extracted from notebook: processing and training
    # (Insert your specific Random Forest logic here)
    st.info("The model is ready for training on the features: Pclass, Sex, Age, and Fare.")
