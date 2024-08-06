from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get('value')
    try:
        value = float(data)
        prediction = model.predict(np.array([[value]]))[0]
        return jsonify({'prediction': prediction})
    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter a numerical value.'})

if __name__ == "__main__":
    app.run(debug=True)
