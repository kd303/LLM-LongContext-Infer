import triton_python_backend_utils as pb_utils
from transformers import CLIPTokenizerFast
import numpy as np

# TritonPythonModel class is looked by the Triton server, and it will execute init/execute/finalize method. It does not extend any existing class
# since the Triton C++ server, look to iteract with Python script throug a stub process based on Duck Typing and looks for exact 
# Class name.
class TritonPythonModel:
    def initialize(self, args):
        # Load the Tokenizer once during startup
        self.tokenizer = CLIPTokenizerFast.from_pretrained(
            "openai/clip-vit-base-patch32", 
            use_fast=True
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # 1. Extract raw text from Triton request
            in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT_INPUT")
            # Triton strings are byte-encoded in a numpy object array
            raw_texts = [t.decode('utf-8') for t in in_tensor.as_numpy().tolist()]

            # 2. Tokenize the entire batch at once
            tokens = self.tokenizer(
                raw_texts,
                padding='max_length',
                max_length=77,
                truncation=True,
                return_tensors="np"
            )

            # 3. Create output tensors (using INT32 for model compatibility)
            out_ids = pb_utils.Tensor("INPUT_IDS", tokens['input_ids'].astype(np.int32))
            out_mask = pb_utils.Tensor("ATTN_MASK", tokens['attention_mask'].astype(np.int32))

            responses.append(pb_utils.InferenceResponse(output_tensors=[out_ids, out_mask]))
            
        return responses