
from model import Qwen2VLForConditionalGeneration
from model.src.processing_qwen2_vl import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import torch
import time, argparse

def configure_DART(model, config):

    if config['Sparse']:
        model.config.DART_config = config

    else:
        model.config.DART_config = None

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse', action="store_true", help="Using sparse")

    args = parser.parse_args()

    # model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"
    model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-7B-Instruct"
    device = 'cuda'
    
    processor = Qwen2VLProcessor.from_pretrained(
        model_path, 
    )
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "../chrome.png",
                # "resized_height": 28 * 30,
                # "resized_width": 28 * 30,
            },
            {
                "type": "image",
                "image": "../coursera-1.jpg",
                # "resized_height": 28 * 30,
                # "resized_width": 28 * 30,
            },
            {
                "type": "image",
                "image": "../coursera-2.jpg",
                # "resized_height": 28 * 30,
                # "resized_width": 28 * 30,
            },
            {"type": "text", "text": "Describe this image in more than 200 words."},
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
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    )

    Sparse=args.sparse
    pruned_layer=2
    image_token_start_index=0
    image_token_length=0
    max_num_trunction=0
    reduction_ratio=0.778
    pivot_image_token=4
    pivot_text_token=4

    DART_config = {
          "Sparse": Sparse,
          "K": pruned_layer,
          "image_token_start_index": image_token_start_index,
          "image_token_length": image_token_length,
          "max_num_trunction": max_num_trunction,
          "reduction_ratio": reduction_ratio,
          "pivot_image_token": pivot_image_token,
          "pivot_text_token": pivot_text_token,
    }

    configure_DART(model, DART_config) # HACK
    # print(model.config)
    print('Sparse', Sparse)
    torch.cuda.synchronize()  # Ensure all computations are finished
    start_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=200)

        generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

    torch.cuda.synchronize()
    end_time = time.time()
    generated_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
    total_time = end_time - start_time
    tps = generated_tokens / total_time


    print(f"Model - TPS: {tps:.2f}, Time: {total_time:.2f}s, generated tokens: {generated_tokens}")
