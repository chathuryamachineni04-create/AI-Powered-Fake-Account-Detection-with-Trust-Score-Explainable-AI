import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

# Load dataset
df = pd.read_csv("dataset.csv")

# Target column (1 = Fake, 0 = Real)
y = df['fake']

# Preprocess
X, scaler = preprocess_data(df)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model trained and saved successfully!")