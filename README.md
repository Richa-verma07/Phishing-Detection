# Phishing-Detection

This project is a machine learning-based web application designed to detect whether a given URL is legitimate or phishing. It uses a Random Forest Classifier trained on URL features to classify suspicious web links in real time.

🚀 Features
1. URL-based phishing detection
2. Extracts important features (e.g., presence of @, URL length, https usage, etc.)
3. Trained using a labeled dataset
4. Simple and responsive web interface using Flask
5. Displays prediction result as "Legitimate" or "Phishing" with color-coded output

🧠 Tech Stack
Python 3
Scikit-learn (for model training)
Flask (for web app)
HTML/CSS (for UI)
Pandas & NumPy (for data processing)
Pickle (for saving the model)

📂 Project Structure
phishing-detection/
│
├── phishing_model.pkl           # Trained model file
├── features.py                  # Feature extraction script
├── app.py                       # Flask app backend
├── templates/
│   └── index.html               # Web UI
├── static/                      # (Optional: for CSS/images)
└── phishing_dataset.csv         # Dataset used for training

🧪 How to Run
1. Clone the repo

2. Install dependencies:
pip install flask scikit-learn pandas

3. Run the Flask app:
python app.py

4. Visit http://localhost:5000 in your browser.

5. Enter any URL to check its status.
