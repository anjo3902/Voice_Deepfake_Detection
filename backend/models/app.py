import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

from models.enhanced_detector import EnhancedTTSDetector
from models.aasist import AASIST

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(backend_dir, 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'flac', 'mp3', 'm4a', 'ogg'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[OK] Device: {device}")

model = None
model_path = os.path.join(backend_dir, 'checkpoints', 'zero_download_model.pth')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    global model
    try:
        print(f"\n[OK] Loading AASIST base model...")
        base_model = AASIST()
        
        print(f"[OK] Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        base_model.load_state_dict(checkpoint['model'])
        base_model = base_model.to(device)
        base_model.eval()
        
        print("[OK] Initializing enhanced detector...")
        model = EnhancedTTSDetector(base_model)
        
        print("[OK] Model loaded successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_audio(file_path):
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # Ensure minimum length
        min_length = 16000  # 1 second
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)))
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
        
        return audio_tensor, audio, sr
    except Exception as e:
        print(f"[ERROR] Error processing audio: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\n[OK] Processing file: {filename}")
        
        # Process audio
        audio_tensor, audio_np, sr = process_audio(filepath)
        
        # Get prediction from enhanced detector
        result = model.predict(audio_tensor, audio_np, sr)
        
        # Clean up
        os.remove(filepath)
        
        print(f"[OK] Prediction: {result['prediction']} (Confidence: {result['confidence']*100:.2f}%)")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[ERROR] Prediction error: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  VOICE DEEPFAKE DETECTOR API")
    print("="*50)
    print(f"[OK] Python version: {sys.version}")
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[OK] CUDA device: {torch.cuda.get_device_name(0)}")
    
    print(f"\n[OK] Model path: {model_path}")
    print(f"[OK] Model exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        print(f"[OK] Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Load model
    if not load_model():
        print("\n[ERROR] Failed to load model. Exiting...")
        sys.exit(1)
    
    # SSL configuration
    cert_file = os.path.join(os.path.dirname(backend_dir), 'certificates', 'cert.pem')
    key_file = os.path.join(os.path.dirname(backend_dir), 'certificates', 'key.pem')
    
    print(f"\n[OK] Certificate: {cert_file}")
    print(f"[OK] Key: {key_file}")
    print(f"[OK] Cert exists: {os.path.exists(cert_file)}")
    print(f"[OK] Key exists: {os.path.exists(key_file)}")
    
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        print("\n[ERROR] SSL certificates not found!")
        print("[!] Run: openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("  SERVER STARTING")
    print("="*50)
    print("[HTTPS] Server running on port 5000")
    print("[OK] Endpoint: https://localhost:5000/predict")
    print("[OK] Health check: https://localhost:5000/health")
    print("\n[!] You'll see a security warning in your browser")
    print("[!] This is normal for self-signed certificates")
    print("[!] Click 'Advanced' and 'Proceed to localhost'")
    print("="*50 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        ssl_context=(cert_file, key_file)
    )
