import json
import os
import torch
from tqdm import tqdm
from model import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  

# 1. Setup
device      = 'cuda'
model_path  = "/home/syc/intern/wanshan/Qwen2VL-Resampler-Finetune/output/resampler_7b_retain_ratio_1"
max_pixels  = 1680 * 28 * 28
# vis_dir     = "./visualize_imgs"
# os.makedirs(vis_dir, exist_ok=True)
retain_ratio = 0.9418 # correspond to uimask_ratio = 0.9
processor = Qwen2VLProcessor.from_pretrained(model_path, max_pixels=max_pixels)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map=device,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    retain_ratio=retain_ratio
)

model_response_path = "/home/syc/intern/wanshan/Qwen2-VL/agent_tasks/ScreenSpot/sim_prunelayer_0-04-25/screenspot_sim_qwen2vl-7b_max_pixels_1680-prune_layer-0-retain_ratio-0.9418-web.json"

model_response_name = os.path.basename(model_response_path)

# 2. Load your JSON file
with open(model_response_path, "r") as f:
    data = json.load(f)

# 3. Iterate, run the model, collect mask, and attach it
for inst in tqdm(data[:100], desc="Generating select_masks"):
    # build messages for this instance
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": inst["img_path"],
                
            },
            {"type": "text", "text": inst["text"]},
        ],
    }]

    # 3a. Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 3b. Forward through your ViT to get select_mask
    #    (assuming your vit call returns (visual_tokens, select_mask))
    _, select_mask = model.visual(inputs["pixel_values"], inputs["image_grid_thw"])

    # 3c. Convert mask to list and store
    inst["select_mask"] = select_mask.cpu().tolist()

# 4. Write out a new JSON with masks embedded
with open(model_response_name, "w") as f:
    json.dump(data, f, indent=2)

print("All done ➜ data_with_masks.json created with a `select_mask` field in each instance.")
