from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from processing_qwen2_vl import Qwen2VLProcessor
from transformers import Qwen2VLConfig

# from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
# from configuration_qwen2_vl import Qwen2VLConfig
from qwen_vl_utils import process_vision_info
import time, torch

def get_selected_mask(input_ids, config):
    select_mask = torch.ones_like(input_ids, device=input_ids.device, dtype=input_ids.dtype) # select all tokens
    n_image_tokens = (input_ids == config.image_token_id).sum().item() # image_token_id : 151655
    # print(n_image_tokens)
    vision_start = config.vision_start_token_id # <vision_start>
    vision_end = config.vision_end_token_id
    vision_start_indices = (input_ids[0] == vision_start).nonzero(as_tuple=True)
    vision_start_indices = vision_start_indices[0].item()

    vision_end_indices = (input_ids[0] == vision_end).nonzero(as_tuple=True)
    vision_end_indices = vision_end_indices[0].item()
    # print(f"vision start idx {vision_start_indices}, vision end idx {vision_end_indices}, total {vision_end_indices - vision_start_indices -1}")
    select_mask[:, vision_start_indices+512:vision_end_indices] = 0
    # print(select_mask.sum())
    return select_mask

if __name__=="__main__":
    model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"
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
                "resized_height": 28 * 30,
                "resized_width": 28 * 30,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
    ]

    config = Qwen2VLConfig.from_pretrained(model_path)

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
        attn_implementation="flash_attention_2",
        device_map=device,
    ).eval()
    
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