
from qwen_vl_utils import process_vision_info
import time, torch, re, argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rgb", action="store_true", help="Implement RGB based selection method on model"
    )

    parser.add_argument(
        "--showui", action="store_true"
    )

    parser.add_argument(
        '--sim', action="store_true", help="Implement similarity based on model"
    )
    parser.add_argument(
        '--dart', action='store_true', 
    )

    parser.add_argument(
        '--prune_layer', type=int, help='starting pruning from this layer'
    )

    parser.add_argument(
        '--print_tflops', action='store_true',
    )
    args = parser.parse_args()
    
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "./screenspot_mobile.png",
                    },
                    # {
                    #     "type": "image",
                    #     "image": "./screenspot_mobile_1.png",
                    # },
                    # {
                    #     "type": "image",
                    #     "image": "./coursera.png",
                    # },
                    # {
                    #     "type": "image",
                    #     "image": "./coursera-1.jpg",
                    # },
                    # {
                    #     "type": "image",
                    #     "image": "./coursera-2.jpg",
                    # },
                    # {
                    #     "type": "image",
                    #     "image": "./book.jpg",
                    # },
                    {
                        "type": "text",
                        "text": "Describe these images in details (more than 200 words).",
                    },
                ],
            }
        ]

    model_path = "/data/data1/syc/intern/wanshan/models/Qwen2-VL-7B-Instruct" # 7B
    device = "cuda"
    min_pixels = 256 * 28 * 28
    max_pixels = 12800 * 28 * 28

    if args.rgb: # uimask prune layer
        print("Testing UIGRAPH Inference Speed ...")
        from Qwen2VL_uigraph.model_prunelayer import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

        # 1. Screenshot -> Graph
        uigraph_train = True  # Enable ui graph during training
        uigraph_test = True  # Enable ui graph during inference
        uigraph_diff = 1  # Pixel difference used for constructing ui graph
        uigraph_rand = False  # Enable random graph construction
        # 2. Graph -> Mask
        uimask_pre = True  # Prebuild patch selection mask in the preprocessor (not in model layers) for efficiency
        uimask_ratio = (
        1 # Specify the percentage of patch tokens to skip per component
        )
        uimask_rand = (
            False  # Enable random token selection instead of uniform selection
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            prune_layer = args.prune_layer, # enable | disable uimask 
            print_tflops = args.print_tflops # print total tflops at each token generation
        )
        
       

        ratios = [0, 0.2, 0.4, 0.6, 0.8, 1]
        for ratio in ratios:
            print("Ratio", ratio)
            processor = Qwen2VLProcessor.from_pretrained(
                model_path,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                uigraph_train=uigraph_train,
                uigraph_test=uigraph_test,
                uigraph_diff=uigraph_diff,
                uigraph_rand=uigraph_rand,
                uimask_pre=uimask_pre,
                uimask_ratio=ratio,
                uimask_rand=uimask_rand,
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
            )
            inputs = inputs.to(device)
            visual_tokens = inputs['pixel_values'].shape[0] // 4
            dropped_visual_tokens = inputs['input_ids'].shape[1] - inputs['select_mask'].sum()
            # input_ids shape [bsz, seq_len, dim], seq_len = text + visual 
            # select_mask : [bsz, seq_len], select_mask.sum() = text + reduced_visual 
            # -> input_ids.shape[1] - select_mask.sum() = # of dropped visual tokens 
            print("Number of Visual Tokens :", visual_tokens - dropped_visual_tokens)
            # Timed inference
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times = []
            tps_all = []
            for i in range(10):
                start_event.record()
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1)
                end_event.record()
                elapsed_time = start_event.elapsed_time(end_event)
                # print(f"\nElapsed Time {i+1} : {elapsed_time:.2f} ms")
                times.append(elapsed_time)

                num_generated_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
                
                tps = num_generated_tokens * 1000 / elapsed_time
                tps_all.append(tps)
                print(
                    f"Generated_tokens_num : {num_generated_tokens}, TPS : {tps}"
                )
                
            torch.cuda.synchronize()
            print(f"Average Elapsed Time : ", sum(times) / len(times))

            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"Allocated memory: {allocated:.2f} MB")
            print(f"Reserved memory: {reserved:.2f} MB")
            torch.cuda.empty_cache()

    elif args.sim: # sim based
        print("Testing SIM Inference Speed ...")
        from Qwen2VL_sim.model_prunelayer import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        for retain_ratio in [1, 0.901, 0.792, 0.696, 0.603, 0.522]:

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2",
                prune_layer = args.prune_layer, # enable | disable uimask 
                retain_ratio = retain_ratio,
                print_tflops = args.print_tflops # print total tflops at each token generation
            )

            processor = Qwen2VLProcessor.from_pretrained(
                model_path, 
                min_pixels=min_pixels,
                max_pixels=max_pixels,
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
            )

            inputs = inputs.to(device)
            visual_tokens = inputs['pixel_values'].shape[0] // 4
            print("Number of Visual Tokens :", round(visual_tokens * retain_ratio))
                
            # Timed inference
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times = []
            tps_all = []
            for i in range(10):
                start_event.record()
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1)
                end_event.record()
                elapsed_time = start_event.elapsed_time(end_event)
                # print(f"\nElapsed Time {i+1} : {elapsed_time:.2f} ms")
                times.append(elapsed_time)

                num_generated_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
                
                tps = num_generated_tokens * 1000 / elapsed_time
                tps_all.append(tps)
                print(
                    f"Generated_tokens_num : {num_generated_tokens}, TPS : {tps}"
                )
                
            torch.cuda.synchronize()
            print(f"Average Elapsed Time : ", sum(times) / len(times))


            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"Allocated memory: {allocated:.2f} MB")
            print(f"Reserved memory: {reserved:.2f} MB")

            del model
            del processor
            del inputs
            del generated_ids
            torch.cuda.empty_cache()

    elif args.showui:
        print("Tesing SHOWUI Inference Speed ...")
        from ShowUI.showui import ShowUIForConditionalGeneration, ShowUIProcessor
        
        # 1. Screenshot -> Graph
        uigraph_train = True  # Enable ui graph during training
        uigraph_test = True  # Enable ui graph during inference
        uigraph_diff = 1  # Pixel difference used for constructing ui graph
        uigraph_rand = False  # Enable random graph construction
        # 2. Graph -> Mask
        uimask_pre = True  # Prebuild patch selection mask in the preprocessor (not in model layers) for efficiency
        uimask_ratio = 1 # Specify the percentage of patch tokens to skip per component
        uimask_rand = False  # Enable random token selection instead of uniform selection
        

        lm_qwen_layer = 28 # LLM Decoder layers
        def parse_layer_type(str_ranges, L=lm_qwen_layer, default=0):
            # 0 is without layer token selection, 1 is with layer token selection. Below we provide examples:
            # [1,28,1] means that all LM layers use token selection; [1,28,0] means that do not.
            # Interleaved layer-wise '[2,2,1],[4,4,1],[6,6,1],[8,8,1],[10,10,1],[12,12,1],[14,14,1],[16,16,1],[18,18,1],[20,20,1],[22,22,1],[24,24,1],[26,26,1]'
            result = [default] * L
            matches = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', str_ranges)
            for start, end, value in matches:
                start, end, value = int(start) - 1, int(end) - 1, int(value)
                if end >= L:
                    end = L - 1
                result[start:end + 1] = [value] * (end - start + 1)
            return result

        lm_skip_layer = f"[{args.prune_layer},28,1]"
        lm_skip_layer = parse_layer_type(lm_skip_layer, 28)

        ratios = [0, 0.2, 0.4, 0.6, 0.8, 1]
        for ratio in ratios:
            model = ShowUIForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                lm_skip_ratio=ratio, 
                lm_skip_layer=lm_skip_layer,
                attn_implementation="flash_attention_2",
            )
            processor = ShowUIProcessor.from_pretrained(
                model_path, 
                min_pixels=min_pixels, max_pixels=max_pixels,
                uigraph_train=uigraph_train, uigraph_test=uigraph_test, uigraph_diff=uigraph_diff, uigraph_rand=uigraph_rand,
                uimask_pre=True, uimask_ratio=ratio, uimask_rand=uimask_rand,
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
            )
            inputs = inputs.to(device)
            visual_tokens = inputs['pixel_values'].shape[0] // 4           
            dropped_visual_tokens = inputs['input_ids'].shape[1] - inputs['select_mask'].sum()
            print("Number of Visual Tokens :", visual_tokens - dropped_visual_tokens)
            # Timed inference
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times = []
            tps_all = []
            for i in range(10):
                start_event.record()
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1)
                end_event.record()
                elapsed_time = start_event.elapsed_time(end_event)
                # print(f"\nElapsed Time {i+1} : {elapsed_time:.2f} ms")
                times.append(elapsed_time)

                num_generated_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
                
                tps = num_generated_tokens * 1000 / elapsed_time
                tps_all.append(tps)
                print(
                    f"Generated_tokens_num : {num_generated_tokens}, TPS : {tps}"
                )
                
            torch.cuda.synchronize()
            print(f"Average Elapsed Time : ", sum(times) / len(times))
            
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"Allocated memory: {allocated:.2f} MB")
            print(f"Reserved memory: {reserved:.2f} MB")

            del model
            del processor
            del inputs
            del generated_ids
            torch.cuda.empty_cache()

    elif args.dart:
        from Qwen2VL_DART.model import Qwen2VLForConditionalGeneration
        from transformers import Qwen2VLProcessor

        def configure_DART(model, config):

            if config["Sparse"]:
                model.config.DART_config = config

            else:
                model.config.DART_config = None

        Sparse = True
        pruned_layer = args.prune_layer # in original paper, prune_layer is set to 2 but to keep it fair with other method we set it to 0
        image_token_start_index = 0
        image_token_length = 0
        max_num_trunction = 0
        pivot_image_token = 4
        pivot_text_token = 4

        for retain_ratio in [1, 0.901, 0.792, 0.696, 0.603, 0.522]:

            DART_config = {
                "Sparse": Sparse,
                "K": pruned_layer,
                "image_token_start_index": image_token_start_index,
                "image_token_length": image_token_length,
                "max_num_trunction": max_num_trunction,
                "reduction_ratio": 1-retain_ratio,
                "pivot_image_token": pivot_image_token,
                "pivot_text_token": pivot_text_token,
            }


            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            ).eval()

            configure_DART(model, DART_config)  # HACK


            processor = Qwen2VLProcessor.from_pretrained(
                model_path, 
                max_pixels=max_pixels
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
            )

            inputs = inputs.to(device)
            visual_tokens = inputs['pixel_values'].shape[0] // 4
            print("Number of Visual Tokens :", round(visual_tokens * retain_ratio))
                
            # Timed inference
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times = []
            tps_all = []
            for i in range(10):
                start_event.record()
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1)
                end_event.record()
                elapsed_time = start_event.elapsed_time(end_event)
                # print(f"\nElapsed Time {i+1} : {elapsed_time:.2f} ms")
                times.append(elapsed_time)

                num_generated_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
                
                tps = num_generated_tokens * 1000 / elapsed_time
                tps_all.append(tps)
                print(
                    f"Generated_tokens_num : {num_generated_tokens}, TPS : {tps}"
                )
                
            torch.cuda.synchronize()
            print(f"Average Elapsed Time : ", sum(times) / len(times))

            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"Allocated memory: {allocated:.2f} MB")
            print(f"Reserved memory: {reserved:.2f} MB")

            del model
            del processor
            del inputs
            del generated_ids

            torch.cuda.empty_cache()
            
    else: # naive Qwen2VL
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        print("Testing Naive Inference Speed ...")

        processor = Qwen2VLProcessor.from_pretrained(
            model_path, 
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2",
        ).eval()

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
        visual_tokens = inputs['pixel_values'].shape[0] // 4
            
        # Timed inference
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        tps_all = []
        for i in range(10):
            start_event.record()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1)
            end_event.record()
            elapsed_time = start_event.elapsed_time(end_event)
            # print(f"\nElapsed Time {i+1} : {elapsed_time:.2f} ms")
            times.append(elapsed_time)

            num_generated_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
            
            tps = num_generated_tokens * 1000 / elapsed_time
            tps_all.append(tps)
            print(
                f"Generated_tokens_num : {num_generated_tokens}, TPS : {tps}"
            )
            
        torch.cuda.synchronize()
        print(f"Average Elapsed Time : ", sum(times) / len(times))

        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"Allocated memory: {allocated:.2f} MB")
        print(f"Reserved memory: {reserved:.2f} MB")




            

        # print("Average TPS:", sum(tps_all) / len(tps_all))
        # elapsed_time = start_event.elapsed_time(end_event)

        # num_generated_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
        # print(
        #     f"Generated_tokens_num : {num_generated_tokens}, TPS : {num_generated_tokens * 1000 / elapsed_time}"
        # )

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