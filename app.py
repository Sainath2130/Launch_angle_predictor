from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained KNN model
model_path = 'models\knn_model.pkl'  
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        
        # Extract parameters
        HitDistance = data['HitDistance']
        ExitVelocity = data['ExitVelocity']
        
        # Prepare input for prediction
        input_data = pd.DataFrame([[HitDistance, ExitVelocity]], columns=['HitDistance', 'ExitVelocity'])
        
        # Perform prediction
        prediction = model.predict(input_data)
        
        return jsonify({'predicted_hit_distance': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)