import joblib
import streamlit as st
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

# Title
st.title('Multiple Regression Model to Predict Price of Home')

# Load model and data
model = joblib.load(r"artifacts\multiple_reg.joblib")  # use raw string or forward slashes
df = pd.read_csv("home_prices.csv")

col1, col2 = st.columns(2)
with col1:
    # Input from user
    area = st.number_input("Enter Area in Sq Feet", value=656)
with col2:
    # Input from user
    nbeds = st.number_input("Enter Number of Bedrooms", value=2)

# Prediction function
def predict(area_sqr_ft):
    prediction = model.predict(np.array([[area_sqr_ft, nbeds]]))
    return round(float(prediction[0]), 2)

# Dashboard display function
def display_dashboard(prediction_value):
    with st.expander("See full data table"):
        st.write(df)

    tab1, tab2 = st.tabs(["Predicted Price in Lakhs", "Plots"])

    with tab1:
        st.subheader(f'Predicted Price in Lakhs is: {prediction_value}')

    with tab2:
        X = df[['area_sqr_ft', 'bedrooms']]
        y = df['price_lakhs']

        # Fit model
        model = LinearRegression()
        model.fit(X, y)

        # Create mesh grid for plotting the plane
        area_range = np.linspace(df['area_sqr_ft'].min(), df['area_sqr_ft'].max(), 10)
        bedrooms_range = np.linspace(df['bedrooms'].min(), df['bedrooms'].max(), 10)
        area_grid, bedrooms_grid = np.meshgrid(area_range, bedrooms_range)
        price_pred = model.intercept_ + model.coef_[0] * area_grid + model.coef_[1] * bedrooms_grid

        # Plotting
        fig = go.Figure()

        # Actual data points
        fig.add_trace(go.Scatter3d(
            x=df['area_sqr_ft'],
            y=df['bedrooms'],
            z=df['price_lakhs'],
            mode='markers',
            marker=dict(size=5, color='blue'),
            name='Actual Data'
        ))

        # Regression surface
        fig.add_trace(go.Surface(
            x=area_grid,
            y=bedrooms_grid,
            z=price_pred,
            name='Regression Plane',
            opacity=0.6,
            colorscale='Viridis'
        ))

        fig.update_layout(
            title="Multiple Linear Regression Plane",
            scene=dict(
                xaxis_title='Area (sq ft)',
                yaxis_title='Bedrooms',
                zaxis_title='Price (Lakhs)'
            ),
            height=700
        )

        st.plotly_chart(fig)

# Trigger prediction and display
if st.button('Predict'):
    prediction = predict(area)
    st.success(f'Predicted Price in Lakhs is: {prediction}')
    display_dashboard(prediction)




