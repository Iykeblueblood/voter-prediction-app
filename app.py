# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from textblob import TextBlob

# --- Page Configuration ---
st.set_page_config(
    page_title="Abia Voter Behavior Predictor",
    page_icon="🗳️",
    layout="wide"
)

# --- Load Assets ---
# Use a try-except block to handle file loading gracefully
try:
    model = joblib.load('voter_model.joblib')
    feature_names = joblib.load('feature_names.joblib')
    df_mock = pd.read_csv("enhanced_mock_data.csv")
except FileNotFoundError:
    st.error("Model or data files not found. Please run the `train_model.py` script first.")
    st.stop()


# --- Sidebar for User Inputs ---
with st.sidebar:
    st.title("🗳️ Prediction Inputs")
    st.markdown("Adjust the sliders to see how different factors might influence the election outcome.")

    # Get a unique list of LGAs for the dropdown
    lga_list = df_mock['LGA'].unique()
    selected_lga = st.selectbox("Select an LGA", lga_list)
    
    st.header("Demographic & Social Factors")
    input_registered_voters = st.number_input("Registered Voters", min_value=20000, max_value=150000, value=75000, step=1000)
    input_avg_age = st.slider("Average Population Age", 30.0, 60.0, 42.5, 0.5)
    input_unemployment = st.slider("Unemployment Rate (%)", 5, 40, 15) / 100.0
    input_education = st.slider("Population with Higher Education (%)", 30, 80, 50) / 100.0
    input_urban = st.slider("Urban Population (%)", 20, 100, 60) / 100.0

    st.header("Social Media Sentiment")
    st.caption("Score from -1 (very negative) to +1 (very positive)")
    input_senti_pdp = st.slider("PDP Sentiment", -1.0, 1.0, 0.1, 0.05)
    input_senti_apc = st.slider("APC Sentiment", -1.0, 1.0, 0.0, 0.05)
    input_senti_lp = st.slider("LP Sentiment", -1.0, 1.0, 0.5, 0.05)
    
    predict_button = st.button("Predict Winner", use_container_width=True)


# --- Main Page Content ---
st.title("Predictive Voter Behavior Analysis for Abia State")
st.markdown("An interactive tool to explore election data and predict outcomes based on various socio-demographic factors.")
st.warning("This is a demonstration using **mock data**. The predictions are for illustrative purposes and do not reflect real-world outcomes.")


# --- Historical Analysis Dashboard ---
st.header(f"📊 Historical Analysis for {selected_lga}")
st.markdown("Analyze past election results and turnout trends for the selected Local Government Area.")

# Filter data for the selected LGA
historical_data = df_mock[df_mock['LGA'] == selected_lga].set_index('Year')

# Create two columns for a cleaner layout
col1, col2 = st.columns([2, 1]) 

with col1:
    # Use Plotly for an interactive historical bar chart
    fig_hist = px.bar(
        historical_data, 
        x=historical_data.index, 
        y=['PDP_Votes', 'APC_Votes', 'LP_Votes'],
        title=f"Vote Counts in {selected_lga}",
        labels={'value': 'Number of Votes', 'Year': 'Election Year', 'variable': 'Party'},
        barmode='group',
        color_discrete_map={'PDP_Votes': '#007bff', 'APC_Votes': '#28a745', 'LP_Votes': '#dc3545'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("Data at a Glance")
    st.dataframe(historical_data[['Registered_Voters', 'PDP_Votes', 'APC_Votes', 'LP_Votes']])


# --- Prediction and Interpretation Section ---
if predict_button:
    st.header("🔮 Prediction Results & Insights")
    
    # Create a DataFrame from the user's input
    input_data = pd.DataFrame({
        'Registered_Voters': [input_registered_voters],
        'Average_Age': [input_avg_age],
        'Unemployment_Rate': [input_unemployment],
        'Education_High_Pct': [input_education],
        'Urban_Pct': [input_urban],
        'Sentiment_PDP': [input_senti_pdp],
        'Sentiment_APC': [input_senti_apc],
        'Sentiment_LP': [input_senti_lp]
    })
    
    # Use the loaded model to make a prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Winner")
        st.success(f"The model predicts **{prediction}** will win in {selected_lga} with these conditions.")

        st.subheader("Prediction Confidence")
        df_proba = pd.DataFrame({
            'Party': model.classes_,
            'Confidence': prediction_proba[0]
        })
        fig_proba = px.bar(df_proba, x='Party', y='Confidence', color='Party', title="Confidence Scores",
                           color_discrete_map={'PDP': '#007bff', 'APC': '#28a745', 'LP': '#dc3545'})
        fig_proba.update_layout(yaxis_title="Probability")
        st.plotly_chart(fig_proba, use_container_width=True)

    with col2:
        st.subheader("Key Influencing Factors")
        st.markdown("What factors drove this prediction? (Based on the model's internal logic)")
        
        try:
            # Get the coefficients for the predicted class
            predicted_class_index = list(model.classes_).index(prediction)
            importances = model.coef_[predicted_class_index]
        except ValueError: # Handles case if a class has no coefficients in a specific model type
            importances = model.coef_[0]

        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Influence': importances
        }).sort_values(by='Influence', ascending=False)
        
        st.write("Top positive factors:")
        st.dataframe(df_importance.head(3))
        
        st.write("Top negative factors:")
        st.dataframe(df_importance.tail(3).sort_values(by='Influence', ascending=True))

# --- Live Sentiment Demo ---
with st.expander("💬 Live Sentiment Analysis Demo"):
    st.markdown("This tool demonstrates how sentiment could be calculated from text in real-time.")
    text_input = st.text_input("Enter a sample sentence:", "The policies from the Labour Party are very promising.")

    if text_input:
        blob = TextBlob(text_input)
        sentiment = blob.sentiment.polarity
        st.metric("Sentiment Score (from -1 to 1)", f"{sentiment:.2f}")