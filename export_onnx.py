import torch
import timm

def export():
    device = torch.device("cpu")
    num_classes = 3
    
    # Instantiate architecture
    model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=num_classes)
    try:
        state_dict = torch.load("models/mobilenetv4_plant_disease.pth", map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Using default initialized weights for export demo: {e}")
        
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    onnx_path = "models/mobilenetv4_plant_disease.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_tensor'],
        output_names=['class_logits'],
        dynamic_axes={'input_tensor': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}}
    )
    print(f"✅ ONNX model successfully exported to: {onnx_path}")

if __name__ == "__main__":
    export()