from flask import Flask, render_template, request
import pickle
import numpy as np
from urllib.parse import urlparse

# Load trained model
with open("phishing_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Function to extract features from URL
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

# Define routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        url = request.form["url"]
        features = np.array(extract_features(url)).reshape(1, -1)
        prediction = model.predict(features)
        result = "Phishing" if prediction[0] == 1 else "Legitimate"

    return render_template("index.html", result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
