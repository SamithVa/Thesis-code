from model_uigraph_edited.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from model_uigraph_edited.processing_qwen2_vl import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import time 
import torch
import re 
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument("--uigraph", action="store_true", help="enable ui graph token pruning")

args = parser.parse_args()

model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"

device = 'cuda'

min_pixel = 2048*28*28
max_pixel = 2048*28*28
# 1. Screenshot -> Graph
uigraph_train = True        # Enable ui graph during training
uigraph_test = True         # Enable ui graph during inference
uigraph_diff = 1            # Pixel difference used for constructing ui graph
uigraph_rand = False        # Enable random graph construction 
# 2. Graph -> Mask 
uimask_pre = True           # Prebuild patch selection mask in the preprocessor (not in model layers) for efficiency
uimask_ratio = 0.8         # Specify the percentage of patch tokens to skip per component
uimask_rand = False         # Enable random token selection instead of uniform selection



messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./chrome.png",
                "min_pixels": min_pixel,
                "max_pixels": max_pixel,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]



def parse_layer_type(str_ranges, L=28, default=0):
    # 0 is without layer token selection, 1 is with layer token selection. Below we provide examples:
    # [1,28,1] means that all LM layers use token selection; [1,28,0] means that do not.
    # Interleaved layer-wise '[2,2,1],[4,4,1],[6,6,1],[8,8,1],[10,10,1],[12,12,1],[14,14,1],[16,16,1],[18,18,1],[20,20,1],[22,22,1],[24,24,1],[26,26,1]'
    result = [default] * L
    matches = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", str_ranges)
    for start, end, value in matches:
        start, end, value = int(start) - 1, int(end) - 1, int(value)
        if end >= L:
            end = L - 1
        result[start : end + 1] = [value] * (end - start + 1)
    return result

if args.uigraph:
    processor = Qwen2VLProcessor.from_pretrained(
        model_path,
        min_pixels= min_pixel,
        max_pixels = max_pixel,
        uigraph_train=uigraph_train, uigraph_test=uigraph_test, uigraph_diff=uigraph_diff, uigraph_rand=uigraph_rand,
        uimask_pre=True, uimask_ratio=uimask_ratio, uimask_rand=uimask_rand,
        use_fast = True
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype = torch.bfloat16,
        attn_implementation="flash_attention_2",
        lm_skip_layer=parse_layer_type("[1,28,0]"),
        lm_skip_ratio=0.2,
        device_map = device,
        prune_layer = 2
    ).eval()
else:
    import transformers
    model = transformers.Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map = device
    ).eval()
    processor = transformers.Qwen2VLProcessor.from_pretrained(
        model_path
    )

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
    # vis_dir="./visualize_imgs" # this folder to save visualization 
).to(device)

with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start 
    tps = generated_ids.shape[1] - inputs["input_ids"].shape[1] / elapsed_time / 1000
    print(f"Elapsed time {elapsed_time} ms, TPS = {tps}")

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, 
)[0]

print(output_text)