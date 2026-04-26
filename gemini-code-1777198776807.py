import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# 1. FIXED DATA LOADING
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        df = pd.read_csv('train.csv')
    except:
        df = pd.read_csv(url)
    
    # Simple preprocessing for the model
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Training & Prediction"])

if app_mode == "Home":
    st.title("🚢 Titanic Survival Analysis")
    st.write("Explore the historic dataset and train a machine learning model to predict survival.")

elif app_mode == "Data Exploration":
    st.title("📊 Data Exploration")
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    st.write("### Survival by Class")
    fig = px.histogram(df, x="Pclass", color="Survived", barmode="group")
    st.plotly_chart(fig)

elif app_mode == "Model Training & Prediction":
    st.title("🤖 Model Training & Prediction")
    
    # Features for the model
    features = ['Pclass', 'Sex', 'Age', 'Fare']
    X = df[features]
    y = df['Survived']

    # 2. ADDED TRAINING BUTTON
    if st.button('🚀 Train Model'):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        model.fit(X, y)
        st.session_state['model'] = model
        acc = accuracy_score(y, model.predict(X))
        st.success(f"Model trained successfully! Accuracy: {acc:.2%}")

    # 3. PREDICTION INTERFACE
    if 'model' in st.session_state:
        st.divider()
        st.subheader("Predict Survival")
        
        col1, col2 = st.columns(2)
        with col1:
            p_class = st.selectbox("Passenger Class", [1, 2, 3])
            gender = st.selectbox("Sex", ["male", "female"])
        with col2:
            age = st.slider("Age", 0, 100, 25)
            fare = st.number_input("Fare", 0.0, 500.0, 32.0)
        
        if st.button("🔮 Predict"):
            gender_val = 0 if gender == "male" else 1
            input_data = np.array([[p_class, gender_val, age, fare]])
            prediction = st.session_state['model'].predict(input_data)
            
            if prediction[0] == 1:
                st.balloons()
                st.success("Result: The passenger likely Survived!")
            else:
                st.error("Result: The passenger likely did NOT survive.")
    else:
        st.info("Click the 'Train Model' button above to enable predictions.")
