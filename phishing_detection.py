import pickle
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# Sample dataset (Replace with a larger dataset)
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

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("phishing_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
