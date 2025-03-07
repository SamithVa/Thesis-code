import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import time


device = 'cuda:3'
model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct_edited_config"

processor = Qwen2VLProcessor.from_pretrained(model_path, use_fast=False)

min_pixel = 256*28*28
max_pixel = 1344*28*28

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
        {"type": "text", "text": "Describe this image. write 500 words"},
    ],
}
]

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
)

inputs = inputs.to(device)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map = device,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).eval()

tps_arr = []
times = []
for i in range(10):
    torch.cuda.synchronize()  # Ensure all computations are finished
    start_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=200)
        generated_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    tps = generated_tokens / total_time
    tps_arr.append(tps)
    times.append(total_time)

print(sum(tps_arr) / len(tps_arr))
print(sum(times) / len(times))



print(f"Model - TPS: {tps:.2f}, Time: {total_time:.2f}s, generated tokens: {generated_tokens}")

