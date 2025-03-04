from modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from transformers import AutoProcessor
from configuration_qwen2_vl import Qwen2VLConfig
from qwen_vl_utils import process_vision_info

if __name__=="__main__":
    model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"
    device = 'cuda'
    processor = AutoProcessor.from_pretrained(
        model_path
    )
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./chrome.png",
                "resized_height": 600,
                "resized_width": 600,
            },
            {"type": "text", "text": "Describe this image."},
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
    # print(inputs)
    config = Qwen2VLConfig.from_pretrained(model_path)
    print(config.vision_config)
    vit = Qwen2VisionTransformerPretrainedModel(config.vision_config).to(device)
    visual_tokens = vit(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
    print(visual_tokens.shape)