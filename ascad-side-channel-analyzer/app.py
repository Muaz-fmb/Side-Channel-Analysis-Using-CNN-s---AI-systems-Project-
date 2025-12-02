import os
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import io
import base64
import warnings

# Suppress TensorFlow and Keras warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store loaded data
model = None
X_attack = None
Y_attack = None
metadata = None

def load_data():
    """Load the model and dataset at startup."""
    global model, X_attack, Y_attack, metadata
    
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'cnn_best_ascad_desync0_epochs75_classes256_batchsize200.h5')
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'ASCAD.h5')
    
    # Load the model
    if os.path.exists(model_path):
        try:
            # Use custom_objects to handle the 'lr' parameter issue
            model = load_model(model_path, custom_objects={}, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print(f"✓ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("  Attempting to load model without compilation...")
            try:
                model = load_model(model_path, compile=False)
                print(f"✓ Model loaded (without compilation) from {model_path}")
            except Exception as e2:
                print(f"✗ Failed to load model: {e2}")
                model = None
    else:
        print(f"✗ Model file not found at {model_path}")
    
    # Load the dataset
    if os.path.exists(dataset_path):
        try:
            with h5py.File(dataset_path, 'r') as f:
                X_attack = np.array(f['Attack_traces/traces'], dtype=np.int8)
                Y_attack = np.array(f['Attack_traces/labels'])
                # Load metadata if available
                if 'Attack_traces/metadata' in f:
                    metadata = f['Attack_traces/metadata'][:]
            print(f"✓ Dataset loaded successfully from {dataset_path}")
            print(f"  X_attack shape: {X_attack.shape}")
            print(f"  Y_attack shape: {Y_attack.shape}")
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            X_attack = None
            Y_attack = None
            metadata = None
    else:
        print(f"✗ Dataset file not found at {dataset_path}")

def plot_trace(trace_data):
    """Create a plot of the trace data and return as base64 image."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(trace_data, linewidth=0.5)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Power Consumption')
    ax.set_title('Power Trace')
    ax.grid(True, alpha=0.3)
    
    # Convert plot to base64 image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)
    
    return img_base64

def plot_predictions(predictions):
    """Create a bar plot of the predictions and return as base64 image."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Get top 20 predictions
    top_indices = np.argsort(predictions)[-20:][::-1]
    top_values = predictions[top_indices]
    
    ax.bar(range(len(top_values)), top_values, color='steelblue')
    ax.set_xlabel('Class (Key Byte Value)')
    ax.set_ylabel('Probability')
    ax.set_title('Top 20 Predictions')
    ax.set_xticks(range(len(top_values)))
    ax.set_xticklabels([str(int(idx)) for idx in top_indices], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Convert plot to base64 image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)
    
    return img_base64

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/get_trace', methods=['POST'])
def get_trace():
    """Get a trace from the dataset by index."""
    try:
        data = request.json
        index = int(data.get('index', 0))
        
        if X_attack is None:
            return jsonify({'error': 'Dataset not loaded'}), 400
        
        if index < 0 or index >= X_attack.shape[0]:
            return jsonify({'error': f'Index out of range. Valid range: 0-{X_attack.shape[0]-1}'}), 400
        
        trace = X_attack[index].astype(float).tolist()
        
        # Create plot
        trace_plot = plot_trace(X_attack[index])
        
        return jsonify({
            'success': True,
            'trace': trace,
            'trace_length': len(trace),
            'trace_plot': f'data:image/png;base64,{trace_plot}',
            'index': index
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction using the model."""
    try:
        data = request.json
        index = int(data.get('index', 0))
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 400
        
        if X_attack is None:
            return jsonify({'error': 'Dataset not loaded'}), 400
        
        if index < 0 or index >= X_attack.shape[0]:
            return jsonify({'error': f'Index out of range. Valid range: 0-{X_attack.shape[0]-1}'}), 400
        
        # Get the trace
        trace = X_attack[index].astype(np.float32)
        
        # Reshape for model input (add batch dimension)
        trace_input = trace.reshape(1, -1, 1)
        
        # Make prediction
        predictions = model.predict(trace_input, verbose=0)
        predictions = predictions[0]  # Get the first (and only) batch
        
        # Create prediction plot
        pred_plot = plot_predictions(predictions)
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions)[-5:][::-1]
        top_5_probs = predictions[top_5_indices].tolist()
        
        # Get the correct label for this trace
        correct_label = int(Y_attack[index])
        
        # Get the predicted label (highest probability)
        predicted_label = int(top_5_indices[0])
        
        # Check if prediction matches the correct label
        is_match = predicted_label == correct_label
        
        return jsonify({
            'success': True,
            'predictions': predictions.tolist(),
            'top_5_classes': [int(idx) for idx in top_5_indices],
            'top_5_probabilities': top_5_probs,
            'prediction_plot': f'data:image/png;base64,{pred_plot}',
            'index': index,
            'correct_label': correct_label,
            'predicted_label': predicted_label,
            'is_match': is_match,
            'match_confidence': float(predictions[predicted_label])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/dataset_info', methods=['GET'])
def dataset_info():
    """Get information about the dataset."""
    if X_attack is None:
        return jsonify({'error': 'Dataset not loaded'}), 400
    
    # Convert numpy int64 to Python int for JSON serialization
    num_traces = int(X_attack.shape[0])
    trace_length = int(X_attack.shape[1])
    num_classes = int(Y_attack.max() + 1) if Y_attack is not None else 256
    
    return jsonify({
        'num_traces': num_traces,
        'trace_length': trace_length,
        'num_classes': num_classes
    })

if __name__ == '__main__':
    print("=" * 60)
    print("ASCAD Power Trace Predictor - Starting...")
    print("=" * 60)
    
    # Load data at startup
    load_data()
    
    print("=" * 60)
    print("Flask app is starting...")
    print("=" * 60)
    
    # Run the Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)
