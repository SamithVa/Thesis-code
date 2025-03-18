import torch
# from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from processing_qwen2_vl import Qwen2VLProcessor
from configuration_qwen2_vl import Qwen2VLConfig
from qwen_vl_utils import process_vision_info
import time


device = 'cuda'
config_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct-Merge-4/config.json"

save_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-Merge-4"
config = Qwen2VLConfig.from_json_file(config_path)
print('loading model')
model = Qwen2VLForConditionalGeneration(config)
print('saving model')

model.save_pretrained(save_path)

print("saving processor")
processor = Qwen2VLProcessor.from_pretrained("/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct")
processor.save_pretrained(save_path)
