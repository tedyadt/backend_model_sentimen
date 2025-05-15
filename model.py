from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

dataset = pd.read_csv('preprocessedreal.csv') .head(50)#load csv

# Load model dan komponen
vectorizer = joblib.load('vectorizer.pkl')
chi2_selector = joblib.load('chi2_selector.pkl')
svm_model = joblib.load('svm_model.pkl')

def preprocess(text):
    tokens = text.lower().split()
    return tokens   

@app.route('/')
def index():
    return "Sentiment API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get('text', '')

    tokens = preprocess(input_text)
    input_tfidf = vectorizer.transform([' '.join(tokens)])
    input_selected = chi2_selector.transform(input_tfidf)
    prediction = svm_model.predict(input_selected)[0]

    return jsonify({'prediction': prediction})

@app.route('/dataset', methods=['GET'])
def get_dataset():
    # Convert dataset ke list of dict
    dataset_json = dataset.to_dict(orient='records')
    return jsonify(dataset_json)
@app.route('/dataset/before', methods=['GET'])
def get_dataset_before():
    selected_columns = dataset[['full_text']]
    return jsonify(selected_columns.to_dict(orient='records'))

@app.route('/dataset/after', methods=['GET'])
def get_dataset_after():
    selected_columns = dataset[['text_clean', 'stemmed_text', 'polarity']]
    return jsonify(selected_columns.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
