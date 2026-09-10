import torch
import torchvision.models as models
import os

def convert_to_onnx():
    print("Initialize ONNX Export Pipeline...")

    # 1. Define Model Architecture and Load Weights
    # NOTE: Ensure the num_classes matches the exact number of diseases in your dataset
    num_classes = 15  
    
    # Assuming MobileNet architecture based on your project description. 
    # If using a custom architecture, import and instantiate it here instead.
    model = models.mobilenet_v3_large(num_classes=num_classes) 
    
    # Load your trained weights (.pth or .pt file)
    model_weights_path = "models/best_model.pth" # Update this path to where your .pth file is saved
    
    if not os.path.exists(model_weights_path):
        print(f"Error: Could not find weights file at {model_weights_path}")
        print("Please place your trained .pth file in the correct folder.")
        return

    # Load state dict and set to evaluation mode (CRITICAL for inference)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"PyTorch model weights loaded successfully from {model_weights_path}")

    # 2. Create a Dummy Input Tensor
    # B2B Edge cameras usually send 224x224 RGB images. 
    # The batch size is 1 for single-image API requests.
    dummy_input = torch.randn(1, 3, 224, 224)

    # 3. Export to ONNX
    onnx_file_path = "models/phytovision_v1.onnx"
    
    print("Exporting graph to ONNX format...")
    torch.onnx.export(
        model,                       # The loaded PyTorch model
        dummy_input,                 # The dummy tensor to trace the graph
        onnx_file_path,              # Where to save the output
        export_params=True,          # Store the trained parameter weights inside the ONNX file
        opset_version=12,            # Standard opset for maximum hardware compatibility
        do_constant_folding=True,    # Optimizes the graph by folding constant nodes
        input_names=['input_image'], # Standardized input name for FastAPI parsing
        output_names=['disease_probabilities'], # Standardized output name
        dynamic_axes={
            'input_image': {0: 'batch_size'},         # Allows FastAPI to process multiple images if needed later
            'disease_probabilities': {0: 'batch_size'}
        }
    )

    print(f"Success! Model exported to {onnx_file_path}")
    print("This file is now ready to be mounted into the FastAPI inference engine.")

if __name__ == "__main__":
    convert_to_onnx()