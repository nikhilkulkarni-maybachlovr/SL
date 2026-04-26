import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="Disaster Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Title and Introduction
st.title("🚢 Machine Learning from Disaster")
st.markdown("""
    This application predicts survival outcomes based on passenger data. 
    Explore the data, visualize key trends, and test the machine learning model.
""")

# Sidebar for navigation and parameters
st.sidebar.header("Settings")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "Data Exploration", "Model Training", "Make Predictions"])

# Loading Data
@st.cache_data
def load_data():
    # Replace with your actual data loading logic from the notebook
    # df = pd.read_csv('train.csv')
    # return df
    pass

df = load_data()

if app_mode == "Home":
    st.subheader("Welcome to the Disaster Survival Analysis")
    st.image("https://images.unsplash.com/photo-1500077423678-25eead48513a", caption="Ship at Sea")
    st.write("Use the sidebar to explore dataset insights or predict survival probabilities.")

elif app_mode == "Data Exploration":
    st.subheader("📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Dataset Preview")
        st.dataframe(df.head())
    
    with col2:
        st.write("### Data Statistics")
        st.write(df.describe())

    st.write("### Survival Correlation by Feature")
    feature = st.selectbox("Select Feature to Compare with Survival", ["Pclass", "Sex", "Embarked"])
    fig = px.histogram(df, x=feature, color="Survived", barmode="group", title=f"Survival Count by {feature}")
    st.plotly_chart(fig)

elif app_mode == "Model Training":
    st.subheader("🤖 Model Performance")
    # This section replicates the ML training logic from your notebook
    st.info("The model is trained using a Random Forest Classifier.")
    # Show metrics like Accuracy, Confusion Matrix, and Feature Importance
    # st.write(f"Model Accuracy: {accuracy:.2f}")

elif app_mode == "Make Predictions":
    st.subheader("🔮 Predict Survival")
    
    with st.form("prediction_form"):
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 100, 25)
        fare = st.number_input("Fare Paid", 0.0, 500.0, 32.0)
        
        submit = st.form_submit_button("Predict")
        
        if submit:
            # logic to transform inputs and call model.predict()
            st.success("Prediction complete!")
            # st.write(f"The passenger would have: {'Survived' if prediction[0] == 1 else 'Perished'}")