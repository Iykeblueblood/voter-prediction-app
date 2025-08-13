import pandas as pd
import numpy as np

print("Generating enhanced mock data with CAUSAL links between sentiment and votes...")

# List of the 17 LGAs in Abia State
lgas = [
    "Aba North", "Aba South", "Arochukwu", "Bende", "Ikwuano", "Isiala Ngwa North",
    "Isiala Ngwa South", "Isuikwuato", "Obi Ngwa", "Ohafia", "Osisioma Ngwa", "Ugwunagbo",
    "Ukwa East", "Ukwa West", "Umuahia North", "Umuahia South", "Umunneochi"
]

data = []

# --- NEW: Define a base multiplier for sentiment's effect ---
# This controls how much sentiment affects the final vote count.
SENTIMENT_VOTE_MULTIPLIER = 10000 

# Generate data for both election years
for lga in lgas:
    for year in [2019, 2023]:
        
        # --- MODIFIED VOTE GENERATION WITH CAUSAL SENTIMENT LINK ---

        # 1. Generate random base data as before
        pdp_base = np.random.randint(12000, 22000)
        apc_base = np.random.randint(10000, 20000)
        sentiment_pdp = round(np.random.uniform(-0.5, 0.5), 2)
        sentiment_apc = round(np.random.uniform(-0.4, 0.4), 2)
        
        # 2. Calculate a "Sentiment Boost" based on the sentiment score
        pdp_sentiment_boost = int(sentiment_pdp * SENTIMENT_VOTE_MULTIPLIER)
        apc_sentiment_boost = int(sentiment_apc * SENTIMENT_VOTE_MULTIPLIER)
        
        # 3. Add the boost to the base votes
        pdp_votes = max(0, pdp_base + pdp_sentiment_boost)
        apc_votes = max(0, apc_base + apc_sentiment_boost)
        
        # 4. Handle LP logic separately for 2023
        if year == 2023:
            lp_base = np.random.randint(15000, 25000)
            sentiment_lp = round(np.random.uniform(0.1, 0.8), 2) # LP sentiment is generally positive in 2023 mock
            lp_sentiment_boost = int(sentiment_lp * SENTIMENT_VOTE_MULTIPLIER)
            lp_votes = max(0, lp_base + lp_sentiment_boost)
        else: # year == 2019
            lp_votes = 0
            sentiment_lp = 0

        # --- END OF MODIFIED LOGIC ---

        data.append({
            "Year": year,
            "LGA": lga,
            "Registered_Voters": np.random.randint(40000, 100000),
            "PDP_Votes": pdp_votes,
            "APC_Votes": apc_votes,
            "LP_Votes": lp_votes,
            "Average_Age": round(np.random.uniform(35, 50), 1),
            "Unemployment_Rate": round(np.random.uniform(0.10, 0.35), 2),
            "Education_High_Pct": round(np.random.uniform(0.4, 0.75), 2),
            "Urban_Pct": round(np.random.uniform(0.3, 0.9), 2),
            "Sentiment_PDP": sentiment_pdp,
            "Sentiment_APC": sentiment_apc,
            "Sentiment_LP": sentiment_lp,
        })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("enhanced_mock_data.csv", index=False)

# Verification step remains the same
def get_winner(row):
    votes = {'PDP': row['PDP_Votes'], 'APC': row['APC_Votes'], 'LP': row['LP_Votes']}
    return max(votes, key=votes.get)
df['Winning_Party'] = df.apply(get_winner, axis=1)
print("\nVerification of winners in the generated data:")
print(df['Winning_Party'].value_counts())

print("\nData saved successfully to enhanced_mock_data.csv")