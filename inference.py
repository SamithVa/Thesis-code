from Qwen2VL.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from Qwen2VL.processing_qwen2_vl import Qwen2VLProcessor

from ShowUI.showui.modeling_showui import ShowUIForConditionalGeneration
from ShowUI.showui.processing_showui import ShowUIProcessor

from qwen_vl_utils import process_vision_info
import time, torch, re, argparse

from transformers import TextIteratorStreamer
from threading import Thread


# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

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

class MyTextStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.generated_ids = None  # We'll store a tensor here

    def put(self, token_ids):
        # Ensure token_ids is 2D (batch_size, sequence_length)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        token_ids = token_ids.detach().clone()  # detach to avoid issues
        
        if self.generated_ids is None:
            self.generated_ids = token_ids
        else:
            # Concatenate along the sequence dimension (dim=1)
            self.generated_ids = torch.cat([self.generated_ids, token_ids], dim=1)
        super().put(token_ids)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uigraph", action="store_true", help="Implement uigraph on model"
    )
    parser.add_argument(
        "--use_cache", action="store_true", help="Implement uigraph on model"
    )
    args = parser.parse_args()

    model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-2B-Instruct"
    # model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-7B-Instruct" # 7B
    device = "cuda"

    min_pixels = 1344 * 28 * 28
    max_pixels = 1680 * 28 * 28
    # 1. Screenshot -> Graph
    uigraph_train = True  # Enable ui graph during training
    uigraph_test = True  # Enable ui graph during inference
    uigraph_diff = 1  # Pixel difference used for constructing ui graph
    uigraph_rand = False  # Enable random graph construction
    # 2. Graph -> Mask
    uimask_pre = True  # Prebuild patch selection mask in the preprocessor (not in model layers) for efficiency
    uimask_ratio = (
        0.9  # Specify the percentage of patch tokens to skip per component
    )
    uimask_rand = (
        False  # Enable random token selection instead of uniform selection
    )

    if args.uigraph:
        ### ShowUI Model
        lm_skip_ratio = uimask_ratio  # valid if not uimask_pre
        lm_skip_layer = "[1,14,1]"  # [1,28,1] means we apply UI guide token selection from 1-th to 28-th layer (28 is the last layer of Qwen2-VL)

        lm_qwen_layer = 28
        lm_skip_layer = parse_layer_type(lm_skip_layer, 28)

        processor = ShowUIProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            uigraph_train=uigraph_train,
            uigraph_test=uigraph_test,
            uigraph_diff=uigraph_diff,
            uigraph_rand=uigraph_rand,
            uimask_pre=uimask_pre,
            uimask_ratio=uimask_ratio,
            uimask_rand=uimask_rand,
        )

        model = ShowUIForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            lm_skip_ratio=lm_skip_ratio,
            lm_skip_layer=lm_skip_layer,
            use_cache=args.use_cache,
            attn_implementation="flash_attention_2",
        )
    else:
        # processor = Qwen2VLProcessor.from_pretrained(model_path)
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16,
        #     device_map=device,
        #     use_cache=args.use_cache,
        #     attn_implementation="flash_attention_2",
        # )
        lm_skip_ratio = 0  # valid if not uimask_pre
        lm_skip_layer = "[1,28,1]"  # [1,28,1] means we apply UI guide token selection from 1-th to 28-th layer (28 is the last layer of Qwen2-VL)

        lm_qwen_layer = 28
        lm_skip_layer = parse_layer_type(lm_skip_layer, 28)
        processor = ShowUIProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            uigraph_train=uigraph_train,
            uigraph_test=uigraph_test,
            uigraph_diff=uigraph_diff,
            uigraph_rand=uigraph_rand,
            uimask_pre=uimask_pre,
            uimask_ratio=lm_skip_ratio,
            uimask_rand=uimask_rand,
        )

        model = ShowUIForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            lm_skip_ratio=lm_skip_ratio,
            lm_skip_layer=lm_skip_layer,
            use_cache=args.use_cache,
            attn_implementation="flash_attention_2",
        )

    # streaming output

    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "./chrome.png",
                },
                {
                    "type": "image",
                    "image": "./coursera.png",
                },
                {
                    "type": "image",
                    "image": "./coursera-1.jpg",
                },
                {
                    "type": "image",
                    "image": "./coursera-2.jpg",
                },
                {
                    "type": "image",
                    "image": "./book.jpg",
                },
                {
                    "type": "text",
                    "text": "Describe these images in details (more than 500 words).",
                },
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

    # Warm-up phase: Run several iterations to mitigate initialization overhead
    warmup_iterations = 3
    with torch.no_grad():
        for i in range(warmup_iterations):
            _ = model.generate(**inputs, max_new_tokens=200)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    streamer = MyTextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=500)
    
    # Timed inference
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_event.record()

    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            print(new_text, end="", flush=True)
        thread.join()  # Ensure generation thread has finished

        generated_ids = streamer.generated_ids
    end_event.record()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"\nElapsed Time : {elapsed_time:.2f} ms")

    num_generated_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
    print(
        f"Generated_tokens_num : {num_generated_tokens}, TPS : {num_generated_tokens * 1000 / elapsed_time}"
    )

    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :]
    #     for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False,
    # )
    # print("response", output_text)