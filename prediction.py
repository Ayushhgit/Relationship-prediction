import pandas as pd
import numpy as np
import joblib
import os
import sys

CATEGORICAL_COLS = [
    'is_public', 'story_freq', 'city_tier', 'party_freq', 'cafe_freq', 
    'travel_freq', 'gym_freq', 'is_influencer', 'dm_open', 'toxic_ex', 
    'daddy_issues', 'mommy_issues', 'therapy_issues', 'situationship_lore', 
    'soft_launch_history', 'astrology_phase', 'friend_circle_size', 
    'spotify_vibe', 'bio_type'
]

NUMERIC_COLS = [
    'followers', 'following', 'post_count', 'random_chaos'
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'lgbm_bodycount_pipeline.joblib')

def get_user_input():
    print("\n--- Enter User Details for Prediction ---")
    data = {}
    
    # Numeric inputs
    print("\n[Numeric Features]")
    for col in NUMERIC_COLS:
        while True:
            try:
                val = input(f"{col}: ")
                if col == 'random_chaos':
                    data[col] = float(val)
                else:
                    data[col] = int(val)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Categorical inputs
    print("\n[Categorical/Ordinal Features]")
    for col in CATEGORICAL_COLS:
        while True:
            try:
  
                hint = ""
                if 'is_' in col or 'open' in col or 'issues' in col or 'ex' in col:
                    hint = "(0=No, 1=Yes)"
                
                val = input(f"{col} {hint}: ")
                data[col] = int(val)
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")
    
    return data

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("Loading model...")
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    input_data = get_user_input()
    
    df = pd.DataFrame([input_data])
    
    all_cols = NUMERIC_COLS + CATEGORICAL_COLS
 
    print("\nPredicting...")

    try:
        log_pred = pipeline.predict(df)
  
        pred = np.expm1(log_pred)[0]
        
        print(f"\nPredicted Body Count: {pred:.2f}")
        print(f"Rounded Prediction: {int(round(pred))}")
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
