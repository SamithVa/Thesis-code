from Qwen2VL.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from Qwen2VL.processing_qwen2_vl import Qwen2VLProcessor

from qwen_vl_utils import process_vision_info
import time, torch

if __name__=="__main__":
    model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct-Merge-4"
    device = 'cuda'
    processor = Qwen2VLProcessor.from_pretrained(
        model_path
    )
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./chrome.png",
                "resized_height": 28 * 40,
                "resized_width": 28 * 40,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
    ]

    # config = Qwen2VLConfig.from_pretrained(model_path)

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
    # print(inputs['select_mask'].shape)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map=device)
    
    # print(model)
    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    print(generated_ids.shape)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    # start_time = time.time()
    # elapsed_time = time.time() - start_time
    # print(f'elased_time : {elapsed_time}')