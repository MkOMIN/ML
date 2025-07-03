from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Carga modelo y scaler
modelo = joblib.load('modelo_entrenado.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return '✅ API de predicción activa'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = modelo.predict(features_scaled)
        return jsonify({
            'prediction': int(prediction[0]),
            'mensaje': '1 = Win, 0 = Lose'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
