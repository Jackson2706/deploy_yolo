import torch
from torch.onnx import export
from torch.quantization import quantize_dynamic
import onnx
import fire

'''
    For example:
        python script.py --model_path "weights/best.pt" --onnx_filename "best.onnx" --input_shape "(1, 3, 640, 640)" --opset_version 12


'''

def export_yolov5_to_onnx(model_path, onnx_filename, input_shape=(1, 3, 640, 640), opset_version=12):
    """
    Export YOLOv5 PyTorch model to ONNX format.

    Args:
        model_path: Path to YOLOv5 PyTorch model (.pt file).
        onnx_filename: Name for the output ONNX file.
        input_shape: Input shape of the model in the format (batch_size, channels, height, width).
        opset_version: ONNX opset version.
    """
    # Load the YOLOv5 PyTorch model
    model = torch.hub.load('./yolov5', 'custom', source='local', path=model_path, force_reload=True)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Export the model to ONNX
    export(model, dummy_input, onnx_filename, opset_version=opset_version, verbose=True)

    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)

    # Quantize the model dynamically
    quantized_model = quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

    # Evaluate the quantized model to prepare for export
    quantized_model(dummy_input)

    # Convert the quantized PyTorch model to ONNX
    onnx_filename = "quantized_yolov5.onnx"
    torch.onnx.export(quantized_model, dummy_input, onnx_filename, opset_version=12, verbose=True)

    # Load the quantized ONNX model
    quantized_onnx_model = onnx.load(onnx_filename)

    # Optionally, you can optimize the quantized ONNX model here

    # Save the optimized quantized ONNX model
    optimized_quantized_onnx_filename = "optimized_quantized_yolov5.onnx"
    onnx.save(quantized_onnx_model, optimized_quantized_onnx_filename)

if __name__ == "__main__":
    fire.Fire(export_yolov5_to_onnx)
