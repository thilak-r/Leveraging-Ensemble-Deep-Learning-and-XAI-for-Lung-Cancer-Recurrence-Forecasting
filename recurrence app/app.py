import os
import random
import base64
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from timm import create_model
import pydicom
from PIL import Image
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from gradcam import generate_grad_cam, find_target_layer, show_cam_on_image

from pathlib import Path

# Import gemini_utils
from gemini_utils import generate_report_from_results

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'


APP_DIR   = Path(__file__).resolve().parent        # …/recurrence app
PROJECT_ROOT = APP_DIR.parent                     # …/C:\  (project root)

DATASET_PATH   = PROJECT_ROOT / "chest_cancer"
MODEL_SAVE_DIR = PROJECT_ROOT / "Models"

WINDOW_CENTER = -600
WINDOW_WIDTH = 1500

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms
base_input_size = (224, 224)
xception_input_size = (299, 299)
data_transforms = {
    "base": transforms.Compose([
        transforms.Resize(base_input_size), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "Xception": transforms.Compose([
        transforms.Resize(xception_input_size), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Helper functions
def apply_window(img_array, center, width):
    min_val = center - width // 2
    max_val = center + width // 2
    img_array = np.clip(img_array, min_val, max_val)
    if max_val > min_val:
        img_array = (img_array - min_val) / (max_val - min_val)
    else:
        img_array = np.zeros_like(img_array)
    return img_array

def preprocess_dicom(filepath):
    """Preprocess DICOM file to PIL Image"""
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=False)
        img_array = ds.pixel_array.astype(np.float32)
        
        # Apply rescale slope and intercept
        slope = float(ds.RescaleSlope) if 'RescaleSlope' in ds else 1.0
        intercept = float(ds.RescaleIntercept) if 'RescaleIntercept' in ds else 0.0
        if slope != 1.0 or intercept != 0.0:
            img_array = img_array * slope + intercept
        
        # Apply windowing
        img_array_windowed = apply_window(img_array, WINDOW_CENTER, WINDOW_WIDTH)
        
        # Convert to PIL Image
        img_array_clamped = np.clip(img_array_windowed, 0.0, 1.0)
        img_array_uint8 = (img_array_clamped * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_array_uint8).convert('RGB')
        
        return img_pil
    except Exception as e:
        print(f"Error processing DICOM: {e}")
        return None

def pil_to_base64(pil_img):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Initialize models
num_classes = 2
class_names = ['Non-Recurrence', 'Recurrence']

models_definition = {
    "ResNet18": {
        "model": models.resnet18(weights=None), 
        "input_size": (224, 224), 
        "transform": data_transforms["base"]
    },
    "EfficientNet": {
        "model": models.efficientnet_b0(weights=None), 
        "input_size": (224, 224), 
        "transform": data_transforms["base"]
    },
    "DenseNet121": {
        "model": models.densenet121(weights=None), 
        "input_size": (224, 224), 
        "transform": data_transforms["base"]
    },
    "Xception": {
        "model": create_model("xception", pretrained=False, num_classes=1000), 
        "input_size": (299, 299), 
        "transform": data_transforms["Xception"]
    },
    "MobileNetV2": {
        "model": models.mobilenet_v2(weights=None), 
        "input_size": (224, 224), 
        "transform": data_transforms["base"]
    },
}

# Adjust model heads
for name, config in models_definition.items():
    model = config["model"]
    if name == "Xception":
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'fc'):  # ResNet
        if model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):  # EfficientNet, DenseNet, MobileNetV2
        if isinstance(model.classifier, nn.Sequential):
            if len(model.classifier) > 0 and isinstance(model.classifier[-1], nn.Linear):
                if model.classifier[-1].out_features != num_classes:
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif isinstance(model.classifier, nn.Linear):
            if hasattr(model.classifier, 'out_features') and model.classifier.out_features != num_classes:
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Load trained models
trained_models = {}
for model_name, config in models_definition.items():
    model = config["model"]
    load_path = os.path.join(MODEL_SAVE_DIR, f"best_model_{model_name}.pth")
    if os.path.exists(load_path):
        try:
            state_dict = torch.load(load_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            trained_models[model_name] = model
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    else:
        print(f"Model weights not found for {model_name}")

print(f"Loaded {len(trained_models)} models: {list(trained_models.keys())}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not file.filename.lower().endswith('.dcm'):
        return jsonify({'error': 'Please upload a DICOM (.dcm) file'})
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess DICOM
        pil_image = preprocess_dicom(filepath)
        if pil_image is None:
            return jsonify({'error': 'Failed to process DICOM file'})
        
        # Convert original image to base64 for display
        original_image_b64 = pil_to_base64(pil_image)
        
        # Make predictions with all models
        results = []
        votes = {'Non-Recurrence': 0, 'Recurrence': 0}
        
        # Store model prediction details for report generation
        model_results_for_report = []

        for model_name, model in trained_models.items():
            try:
                transform = models_definition[model_name]['transform']
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                input_tensor.requires_grad = True
                
                with torch.set_grad_enabled(True):
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred_idx_tensor = torch.max(probs, dim=1)
                
                pred_class_name = class_names[pred_idx_tensor.item()]
                conf_score = conf.item()
                non_rec_conf = probs[0][0].item()
                rec_conf = probs[0][1].item()
                
                # Count votes
                votes[pred_class_name] += 1
                
                result = {
                    'model': model_name,
                    'prediction': pred_class_name,
                    'confidence': conf_score,
                    'non_recurrence_conf': non_rec_conf,
                    'recurrence_conf': rec_conf,
                    'gradcam': None
                }
                
                # Append to model results for report
                model_results_for_report.append(
                    f"{model_name}: Prediction='{pred_class_name}', Confidence={conf_score:.2f} (Non-Recurrence: {non_rec_conf:.2f}, Recurrence: {rec_conf:.2f})"
                )

                # Generate Grad-CAM if prediction is Recurrence
                if pred_class_name == 'Recurrence':
                    try:
                        target_layer = find_target_layer(model, model_name)
                        if target_layer:
                            grad_cam_heatmap, _ = generate_grad_cam(
                                model, input_tensor, target_layer, pred_idx_tensor.item()
                            )
                            if grad_cam_heatmap is not None:
                                # Create Grad-CAM overlay
                                display_img_np = np.array(pil_image).astype(np.float32) / 255.0
                                grad_cam_overlay = show_cam_on_image(display_img_np, grad_cam_heatmap)
                                grad_cam_pil = Image.fromarray(grad_cam_overlay)
                                result['gradcam'] = pil_to_base64(grad_cam_pil)
                    except Exception as e:
                        print(f"Grad-CAM error for {model_name}: {e}")
                
                results.append(result)
                
            except Exception as e:
                print(f"Prediction error for {model_name}: {e}")
                results.append({
                    'model': model_name,
                    'prediction': 'Error',
                    'confidence': 0.0,
                    'non_recurrence_conf': 0.0,
                    'recurrence_conf': 0.0,
                    'gradcam': None
                })
        
        # Determine majority vote
        if votes['Recurrence'] > votes['Non-Recurrence']:
            majority_vote = 'Recurrence'
        elif votes['Non-Recurrence'] > votes['Recurrence']:
            majority_vote = 'Non-Recurrence'
        else:
            majority_vote = 'Tie'
        
        # Add ensemble result to the report text
        model_results_text = "\n".join(model_results_for_report)
        model_results_text += f"\n\nEnsemble Majority Vote: {majority_vote} (Non-Recurrence: {votes['Non-Recurrence']} votes, Recurrence: {votes['Recurrence']} votes)"

        # Generate reports using gemini_utils
# Try to get patient info from HTML form first
        user_patient_name = request.form.get('patient_name', '').strip()
        user_patient_age = request.form.get('patient_age', '').strip()

        # ... (rest of your prediction logic) ...

        # Use user-provided name/age if available, otherwise fallback to DICOM or "N/A"
        patient_name = "N/A"
        patient_age = "N/A"
        
        if user_patient_name:
            patient_name = user_patient_name
        
        if user_patient_age:
            patient_age = user_patient_age

        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            if not user_patient_name and 'PatientName' in ds: # Only use DICOM if user didn't provide
                patient_name = str(ds.PatientName)
            if not user_patient_age and 'PatientAge' in ds:   # Only use DICOM if user didn't provide
                patient_age = str(ds.PatientAge).replace('Y', '') # Remove 'Y' if present
        except Exception as e:
            print(f"Could not read patient info from DICOM: {e}")

        import datetime
        analysis_datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_data = generate_report_from_results(
            model_results_text, 
            patient_name=patient_name, 
            patient_age=patient_age, 
            analysis_datetime_str=analysis_datetime_str
        )
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'original_image': original_image_b64,
            'results': results,
            'majority_vote': majority_vote,
            'votes': votes,
            'comprehensive_report': report_data.get('comprehensive', 'Error generating comprehensive report.'),
            'synthetic_report': report_data.get('synthetic', 'Error generating synthetic report.')
        })
        
    except Exception as e:
        print(f"General error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)