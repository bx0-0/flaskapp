from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# تحميل النموذج والمشفر
model_path = os.path.join(os.path.dirname(__file__), 'models', 'random_forest_model_compressed.pkl')
model = joblib.load(model_path)

encoder_path = os.path.join(os.path.dirname(__file__), 'models', 'encoder.pkl')
encoder = joblib.load(encoder_path)  # تحميل المشفر

# الصفحة الرئيسية
@app.route('/')
def index():
    return render_template('index.html')

# صفحة التنبؤ (POST method)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate input
        if not all(key in data for key in ('skill1', 'skill2', 'skill3', 'skill4')):
            return jsonify({'error': 'Missing required skills'}), 400

        # Extract inputs
        new_skills = [
            data['skill1'],
            data['skill2'],
            data['skill3'],
            data['skill4']
        ]

        # Encode the inputs
        new_skills_encoded = encoder.transform([new_skills]).toarray()

        # Predict using the model
        predicted_job = model.predict(new_skills_encoded)

        # Return the prediction
        return jsonify(predicted_job[0])

    except ValueError as e:
        return jsonify({'error': f'Input Error: {str(e)}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(host="0.0.0.0")