import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Feature Engineering
    df['ratio'] = df['followers'] / (df['following'] + 1)
    df['activity_rate'] = df['posts'] / (df['account_age'] + 1)
    df['engagement_score'] = df['followers'] / (df['posts'] + 1)
    df['growth_rate'] = df['followers'] / (df['account_age'] + 1)

    features = [
    'account_age',
    'followers',
    'following',
    'posts',
    'bio_length',
    'has_profile_pic',
    'ratio',
    'activity_rate',
    'engagement_score',
    'growth_rate'
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler