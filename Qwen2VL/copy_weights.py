import torch
from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from processing_qwen2_vl import Qwen2VLProcessor 

model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"

target_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct-Merge-4"

processor = Qwen2VLProcessor.from_pretrained(model_path)
# Load source model (pretrained)
source_model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, device_map='cuda:1', torch_dtype=torch.bfloat16)

# Load target model (ensure it has a similar but possibly different architecture)
target_model = Qwen2VLForConditionalGeneration.from_pretrained(target_path, device_map='cuda:0', torch_dtype=torch.bfloat16)  # Change model name if needed

# Load source model's state dictionary
source_state_dict = source_model.state_dict()
target_state_dict = target_model.state_dict()

# Copy matching layers
for name, param in target_state_dict.items():
    if "resampler" not in name:
        if name in source_state_dict and source_state_dict[name].shape == param.shape:
            param.data.copy_(source_state_dict[name])
            print(f"Copied: {name}")
        else:
            print(f"Skipped: {name} (not found in source model or shape mismatch)")

save_path = "/data/data1/syc/intern/wanshan/models/Qwen2VL_copy"
# Save the modified target model
target_model.save_pretrained(save_path)

processor.save_pretrained(save_path)

print("Model weights copied successfully, skipping non-matching layers.")
