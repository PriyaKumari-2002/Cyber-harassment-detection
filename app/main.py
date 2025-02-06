from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load(r'C:\Users\Lenovo\Cyber-harassment-detection\app\models\text_model.pkl')
vectorizer = joblib.load(r'C:\Users\Lenovo\Cyber-harassment-detection\app\models\vectorizer.pkl')


def predict_text(text):
    """Process input text and return model prediction."""
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    confidence = model.predict_proba(text_vec)[0]

    return {
        'text': text,
        'prediction': 'Harassment' if prediction == 1 else 'Non-Harassment',
        'confidence': {
            'Harassment': round(confidence[1], 2),
            'Non-Harassment': round(confidence[0], 2)
        }
    }


@app.route('/')
def home():
    return "Welcome to the Cyber Harassment Detection API!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'texts' not in data:
            raise ValueError("Missing 'texts' field.")

        texts = data['texts']
        if not isinstance(texts, list):
            raise TypeError("'texts' should be a list.")

        # Get predictions for each text
        predictions = [predict_text(text) for text in texts]

        return jsonify({'predictions': predictions})

    except (ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
