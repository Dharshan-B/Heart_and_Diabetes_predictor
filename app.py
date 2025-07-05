from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load models
diabetes_model = pickle.load(open('models/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('models/heart_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    disease_type = data.get("type")
    features = data.get("features")

    prediction = None
    if disease_type == "diabetes":
        prediction = diabetes_model.predict([features])[0]
    elif disease_type == "heart":
        prediction = heart_model.predict([features])[0]

    return jsonify({'result': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
