import torch
import open_clip
from torch import nn

# 1. Load the model
model_name = 'ViT-B-32'
pretrained = 'laion2b_s34b_b79k'
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
model.eval()

# 2. Vision Tower Export
dummy_image = torch.randn(2, 3, 224, 224) # Use batch size > 1 to ensure dynamic axes
vision_path = "openai_clip_vision.onnx"
torch.onnx.export(
    model.visual, 
    dummy_image, 
    vision_path,
    export_params=True,
    opset_version=17, # Latest opset for better TensorRT compatibility
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# 3. Text Tower Wrapper (to include projection)
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
    def forward(self, text):
        return self.clip_model.encode_text(text)

dummy_text = torch.randint(0, 49408, (2, 77))
text_model = TextEncoder(model)
text_path = "openai_clip_text.onnx"
torch.onnx.export(
    text_model,
    dummy_text,
    text_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# 4. Optional: Simplify with onnxsim if installed
try:
    import onnxsim
    import onnx
    print("Simplifying models with onnxsim...")
    for path in [vision_path, text_path]:
        model_onnx = onnx.load(path)
        model_simp, check = onnxsim.simplify(model_onnx)
        if check:
            onnx.save(model_simp, path.replace(".onnx", "_sim.onnx"))
            print(f"Simplified model saved to {path.replace('.onnx', '_sim.onnx')}")
except ImportError:
    print("onnxsim not installed, skipping simplification. Run 'pip install onnx-simplifier' manually.")