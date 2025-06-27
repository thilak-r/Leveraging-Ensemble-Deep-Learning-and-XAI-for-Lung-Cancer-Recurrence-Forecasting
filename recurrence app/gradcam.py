import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2

# Global variables for hooks
features_blobs = []
backward_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.detach().clone())

def hook_backward(module, grad_in, grad_out):
    backward_blobs.append(grad_out[0].detach().clone())

def find_target_layer(model, model_name):
    """Find a suitable last convolutional layer for Grad-CAM."""
    print(f"Finding target layer for {model_name}...")

    if isinstance(model, models.ResNet):
        target = model.layer4[-1].conv3 if hasattr(model.layer4[-1], 'conv3') else model.layer4[-1].conv2
        print(f"Targeting ResNet layer: {type(target).__name__}")
        return target
    
    elif isinstance(model, models.DenseNet):
        # Look backwards within features for the last Conv2d
        for layer in reversed(list(model.features.modules())):
            if isinstance(layer, nn.Conv2d):
                print(f"Targeting DenseNet layer: Last nn.Conv2d in features.")
                return layer
        # Fallback
        target = model.features[-1]
        print(f"Warning: Fallback DenseNet target: {type(target).__name__}")
        return target
    
    elif isinstance(model, models.EfficientNet) or isinstance(model, models.MobileNetV2):
        # Iterate backwards through features to find the last Conv2d
        if hasattr(model, 'features'):
            for layer in reversed(list(model.features.modules())):
                if isinstance(layer, nn.Conv2d):
                    print(f"Targeting {model_name} layer: Last nn.Conv2d in features.")
                    return layer
        # Fallback
        print(f"Warning: No Conv2d found in {model_name}.features. Trying fallback.")
        return model.features[-1]

    elif model_name == "Xception":
        # Target the last Conv2d within the final block
        if hasattr(model, 'block11') and hasattr(model.block11, 'rep'):
            for layer in reversed(list(model.block11.rep.modules())):
                if isinstance(layer, nn.Conv2d):
                    print(f"Targeting Xception layer: Last nn.Conv2d in block11.rep.")
                    return layer
        # Fallback
        print(f"Warning: Could not find conv in Xception block11. Using act4 as fallback.")
        if hasattr(model, 'act4'):
            return model.act4
        else:
            print("Error: Cannot find suitable Xception layer.")
            return None

    else:  # General fallback
        for layer in reversed(list(model.modules())):
            if isinstance(layer, nn.Conv2d):
                print(f"Targeting {model_name} layer: Fallback last nn.Conv2d.")
                return layer

    print(f"Warning: Could not find any Conv2d target layer for {model_name}. Grad-CAM may fail.")
    return None

def generate_grad_cam(model, input_tensor, target_layer, target_class_index):
    """Generate Grad-CAM heatmap."""
    global features_blobs, backward_blobs
    features_blobs = []
    backward_blobs = []
    handle_forward = None
    handle_backward = None
    heatmap_np = None
    calculated_target_index = target_class_index
    
    try:
        # Register hooks
        handle_forward = target_layer.register_forward_hook(hook_feature)
        handle_backward = target_layer.register_full_backward_hook(hook_backward)
        
        # Forward pass
        outputs = model(input_tensor)
        
        if target_class_index is None:
            calculated_target_index = torch.argmax(outputs, dim=1).item()
        else:
            calculated_target_index = target_class_index
        
        # Create one-hot output
        one_hot_output = torch.zeros_like(outputs)
        one_hot_output[0][calculated_target_index] = 1
        
        # Backward pass
        model.zero_grad()
        outputs.backward(gradient=one_hot_output, retain_graph=True)
        
        if not backward_blobs or not features_blobs:
            print("Error: Hooks failed.")
            return None, calculated_target_index
        
        # Get gradients and activations
        gradients = backward_blobs[0].clone()
        activations = features_blobs[0].clone()
        
        # Compute weighted activations
        pooled_gradients = torch.mean(gradients, dim=[2, 3])
        weighted_activations = activations.clone()
        
        for i in range(activations.shape[1]):
            weighted_activations[:, i, :, :] *= pooled_gradients[:, i].unsqueeze(-1).unsqueeze(-1)
        
        # Create heatmap
        heatmap = torch.mean(weighted_activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        
        heatmap_np = heatmap.cpu().numpy()
        
    except Exception as e:
        print(f"Error during Grad-CAM calculation: {e}")
    
    finally:
        # Remove hooks
        if handle_forward:
            handle_forward.remove()
        if handle_backward:
            handle_backward.remove()
        features_blobs = []
        backward_blobs = []
    
    return heatmap_np, calculated_target_index

def show_cam_on_image(img_np, heatmap):
    """Overlay Grad-CAM heatmap on original image."""
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)
    
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Blend images
    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)