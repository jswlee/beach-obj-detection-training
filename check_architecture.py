"""
Detailed script to check YOLO P2 vs P3 architecture
"""
 
import sys
import os
from pathlib import Path
 
# Add the current directory to Python path
sys.path.insert(0, os.getcwd())
 
try:
    from ultralytics import YOLO
 
    model_path = r"training_results\yolov8m_p2_beach_detection_20260401_115424\weights\best.pt"
 
    print(f"Analyzing model: {model_path}")
 
    # Load the model
    model = YOLO(model_path)
 
    # Get detailed model information
    print(f"\n=== MODEL DETAILS ===")
    print(f"Model type: {type(model.model).__name__}")
    print(f"Number of layers: {len(model.model.model)}")
 
    # Look at the model structure more carefully
    model_structure = model.model.model
 
    print(f"\n=== BACKBONE ANALYSIS ===")
 
    # Check the first few layers (backbone)
    backbone_layers = []
    for i, layer in enumerate(model_structure[:12]):  # First 12 layers should include backbone
        layer_name = type(layer).__name__
        print(f"Layer {i}: {layer_name}")
 
        # Look for Conv layers that might have stride info
        if hasattr(layer, 'stride'):
            print(f"  - Stride: {layer.stride}")
            backbone_layers.append((i, layer_name, layer.stride))
        elif hasattr(layer, '__dict__'):
            # Try to find stride in layer attributes
            attrs = layer.__dict__
            if 'stride' in attrs:
                print(f"  - Stride: {attrs['stride']}")
                backbone_layers.append((i, layer_name, attrs['stride']))
            elif 's' in attrs:  # Some layers use 's' for stride
                print(f"  - Stride (s): {attrs['s']}")
                backbone_layers.append((i, layer_name, attrs['s']))
 
    # Check for C2f modules which are common in YOLOv8
    print(f"\n=== C2f MODULES ===")
    for i, layer in enumerate(model_structure):
        if 'C2f' in type(layer).__name__:
            print(f"Layer {i}: {type(layer).__name__}")
            if hasattr(layer, 'cv'):
                print(f"  - Has cv attribute: {type(layer.cv).__name__}")
 
    # Check the model's stride attribute directly
    print(f"\n=== MODEL STRIDE ===")
    if hasattr(model.model, 'stride'):
        print(f"Model stride: {model.model.stride}")
        if isinstance(model.model.stride, (list, tuple)):
            print(f"  - Stride values: {model.model.stride}")
            min_stride = min(model.model.stride)
            print(f"  - Minimum stride: {min_stride}")
 
            if min_stride == 4:
                print("✓ P2 ARCHITECTURE DETECTED (minimum stride = 4)")
            elif min_stride == 8:
                print("✓ P3 ARCHITECTURE DETECTED (minimum stride = 8)")
            else:
                print(f"? Unknown architecture - minimum stride: {min_stride}")
 
    # Check for P2-specific configuration
    print(f"\n=== ARCHITECTURE INDICATORS ===")
 
    # Directory name check
    dir_name = Path(model_path).parent.parent.name
    print(f"Directory name: {dir_name}")
    if 'p2' in dir_name.lower():
        print("→ Directory name indicates P2 architecture")
    elif 'p3' in dir_name.lower():
        print("→ Directory name indicates P3 architecture")
 
    # Check model YAML configuration if available
    if hasattr(model.model, 'yaml'):
        print(f"Model YAML config: {model.model.yaml}")
        if 'backbone' in model.model.yaml:
            backbone = model.model.yaml['backbone']
            for i, layer in enumerate(backbone):
                if isinstance(layer, list) and len(layer) > 3:
                    print(f"Backbone layer {i}: {layer[0]} - args: {layer[1:]}")
 
    # Final conclusion based on directory name (most reliable indicator)
    print(f"\n=== CONCLUSION ===")
    if 'p2' in dir_name.lower():
        print("This model is P2 architecture (based on directory naming)")
        print("P2 architecture uses stride-4 for better small object detection")
    elif 'p3' in dir_name.lower():
        print("This model is P3 architecture (based on directory naming)")
        print("P3 architecture uses stride-8 (standard YOLOv8)")
    else:
        print("Architecture could not be definitively determined")
 
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
 