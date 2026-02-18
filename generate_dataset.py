import numpy as np
import pandas as pd

np.random.seed(42)

N = 15000

data = {
    "followers": np.random.randint(50, 15000, N),
    "following": np.random.randint(50, 3000, N),
    "is_public": np.random.randint(0, 2, N),
    "post_count": np.random.randint(0, 800, N),
    "story_freq": np.random.randint(0, 6, N),
    "city_tier": np.random.choice([1, 2, 3], N, p=[0.4, 0.35, 0.25]),
    "party_freq": np.random.randint(0, 6, N),
    "cafe_freq": np.random.randint(0, 6, N),
    "travel_freq": np.random.randint(0, 6, N),
    "gym_freq": np.random.randint(0, 6, N),
    "is_influencer": np.random.randint(0, 2, N),
    "dm_open": np.random.randint(0, 2, N),
    "toxic_ex": np.random.randint(0, 2, N),
    "daddy_issues": np.random.randint(0, 2, N),
    "mommy_issues": np.random.randint(0, 2, N),
    "therapy_issues": np.random.randint(0, 2, N),
    "situationship_lore": np.random.randint(0, 2, N),
    "soft_launch_history": np.random.randint(0, 2, N),
    "astrology_phase": np.random.randint(0, 2, N),
    "friend_circle_size": np.random.choice([1, 2, 3], N),
    "spotify_vibe": np.random.randint(0, 5, N),
    "random_chaos": np.random.rand(N),

    # NEW: Insta bio category
    # 0 = Chatpate (Medium)
    # 1 = TBSM (Medium-High)
    # 2 = Krishna Sada Sahaye (High)
    "bio_type": np.random.choice([0, 1, 2], N, p=[0.4, 0.35, 0.25])
}

df = pd.DataFrame(data)

def generate_body_count(row):
    score = 0

    # Social exposure
    if row["is_public"]: score += 2
    if row["followers"] > 2000: score += 2
    if row["followers"] > 5000: score += 2
    if row["city_tier"] == 1: score += 2
    if row["is_influencer"]: score += 2

    # Lifestyle chaos
    if row["party_freq"] >= 3: score += 2
    if row["story_freq"] >= 4: score += 1
    if row["dm_open"]: score += 1

    # Lore
    if row["situationship_lore"]: score += 3
    if row["toxic_ex"]: score += 2
    if row["soft_launch_history"]: score += 2

    # Bio impact (your special rule)
    if row["bio_type"] == 0:      # Chatpate
        score += 1
    elif row["bio_type"] == 1:    # TBSM
        score += 2
    elif row["bio_type"] == 2:    # Krishna Sada Sahaye
        score += 3

    # Chaos factor
    score += np.random.randint(0, 4)

    return min(score, 20)

df["body_count"] = df.apply(generate_body_count, axis=1)

df.to_csv("synthetic_bodycount_dataset.csv", index=False)

print(df.head())
