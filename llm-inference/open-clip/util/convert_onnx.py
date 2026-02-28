import torch
import open_clip
from torch import nn

# 1. Load the model
model_name = 'ViT-B-32'
pretrained = 'laion2b_s34b_b79k'
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
model.eval()

# 2. Vision Tower Export
dummy_image = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model.visual, 
    dummy_image, 
    "clip_vision.onnx",
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

dummy_text = torch.randint(0, 49408, (1, 77))
text_model = TextEncoder(model)
torch.onnx.export(
    text_model,
    dummy_text,
    "clip_text.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)