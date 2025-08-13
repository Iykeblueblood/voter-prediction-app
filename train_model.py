# train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

print("Starting model training script...")

# --- Step 1: Load Pre-existing Data ---
try:
    df = pd.read_csv("enhanced_mock_data.csv")
    print("Successfully loaded enhanced_mock_data.csv")
except FileNotFoundError:
    print("Error: enhanced_mock_data.csv not found.")
    print("Please run `create_mock_data.py` first to generate the data file.")
    exit() # Stop the script if data file is missing

# --- Step 2: Preprocess and Feature Engineer ---
print("Preprocessing data...")
# Define the winning party (our target variable)
def get_winner(row):
    votes = {'PDP': row['PDP_Votes'], 'APC': row['APC_Votes'], 'LP': row['LP_Votes']}
    return max(votes, key=votes.get)

df['Winning_Party'] = df.apply(get_winner, axis=1)

# Define features (X) and target (y)
features = [
    'Registered_Voters', 
    'Average_Age',
    'Unemployment_Rate',
    'Education_High_Pct',
    'Urban_Pct',
    'Sentiment_PDP',
    'Sentiment_APC',
    'Sentiment_LP'
]
target = 'Winning_Party'

X = df[features]
y = df[target]

# --- Step 3: Train the Model ---
print("Training the model...")
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X, y)
print("Model training complete.")

# --- Step 4: Save the Model and Feature Names ---
joblib.dump(model, 'voter_model.joblib')
joblib.dump(features, 'feature_names.joblib')
print("Model and feature names have been saved.")
print("Script finished successfully.")