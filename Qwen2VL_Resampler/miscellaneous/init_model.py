import torch
# from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from processing_qwen2_vl import Qwen2VLProcessor
from configuration_qwen2_vl import Qwen2VLConfig
from qwen_vl_utils import process_vision_info
import time


device = 'cuda'
model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"

processor = Qwen2VLProcessor.from_pretrained(model_path, use_fast=False)

min_pixel = 256*28*28
max_pixel = 1344*28*28

messages = [
{
    "role": "user",
    "content": [
        {
            "type": "image",
            "image": "../chrome.png",
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

save_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-Resampler"
inputs = inputs.to(device)
config = Qwen2VLConfig.from_json_file("./config.json")
print('loading model')
model = Qwen2VLForConditionalGeneration(config)
print('saving model')

model.save_pretrained(save_path)
processor.save_pretrained(save_path)
