import re
import pickle
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to extract features from a URL
def extract_features(url):
    parsed_url = urlparse(url)
    
    return [
        len(url),  # URL length
        len(parsed_url.netloc),  # Domain length
        len(parsed_url.path),  # Path length
        1 if "https" in parsed_url.scheme else 0,  # HTTPS presence
        url.count("@"),  # Count '@' symbol
        url.count("-"),  # Count '-' symbol
        url.count("."),  # Count '.' symbol
        url.count("https") + url.count("http"),  # Count http/https occurrences
        url.count("www"),  # Count 'www'
        url.count("?"),  # Count '?' in URL
        url.count("="),  # Count '=' in URL
        url.count("%"),  # Count '%'
        url.count(":"),  # Count ':'
    ]

# Load dataset (Replace with your dataset)
data = [
    ("https://www.google.com", 0),  
    ("http://phishing-site.com/login", 1),  
    ("https://secure-bank.com", 0),
    ("http://fake-bank.com/login.php", 1),
    ("https://www.github.com", 0),
]

df = pd.DataFrame(data, columns=["url", "label"])
df["features"] = df["url"].apply(lambda x: extract_features(x))

# Convert feature lists to numpy array
X = np.array(df["features"].tolist())
y = np.array(df["label"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
with open("phishing_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Function to predict if a URL is phishing
def predict_url(url):
    with open("phishing_model.pkl", "rb") as file:
        model = pickle.load(file)
    features = np.array(extract_features(url)).reshape(1, -1)
    prediction = model.predict(features)
    return "Phishing" if prediction[0] == 1 else "Legitimate"

# Example
test_url = "https://www.google.com"
print(f"URL: {test_url} is {predict_url(test_url)}")
