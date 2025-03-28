import torch
import time
import torch.nn as nn
from typing import List, Optional, Tuple, Union


from model.modeling_qwen2_vl import Qwen2VLModel
from model.configuration_qwen2_vl import Qwen2VLConfig
from model.utils import get_select_mask
import argparse
import re

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


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


def get_rope_index(
    config,
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [3, 4, 5, 6, 7]
            text height position_ids: [3, 4, 5, 6, 7]
            text width position_ids: [3, 4, 5, 6, 7]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def main():
    parser = argparse.ArgumentParser(
        description="Speed test for Qwen2VLModel with variable seq_len, visual ratio, and selection"
    )
    parser.add_argument(
        "--seq_length", type=int, default=8192, help="Total sequence length"
    )
    parser.add_argument(
        "--visual_ratio",
        type=float,
        default=0.8,
        help="Fraction of tokens that are visual tokens (e.g., 0.8 for ~80 tokens out of 100)",
    )
    parser.add_argument(
        "--selection",
        action="store_true",
        help="If set, use the selection module to reduce the number of tokens processed",
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations for timing"
    )
    args = parser.parse_args()

    # Load configuration and adjust settings
    config = Qwen2VLConfig.from_json_file("./config.json")
    # config.num_hidden_layers = 10
    config._attn_implementation = "flash_attention_2"
    if args.selection:
        config.lm_skip_layer = parse_layer_type("[1,28,1]")
        config.lm_skip_ratio = 0.5
    else:
        config.lm_skip_layer = parse_layer_type("[1,28,0]")
    # Optionally, move to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2VLModel(config).to(device, dtype=torch.bfloat16)

    # Define input dimensions.
    batch_size = 1
    seq_len = args.seq_length
    dtype = torch.bfloat16
    # Create random input_ids (dummy input tokens)
    input_ids = torch.randint(
        low=1,
        high=1000,
        size=(batch_size, seq_len),
        device=device,
    )

    # Decide where to insert the visual tokens.
    # Here, we reserve a region starting at index 1000.
    visual_start_index = 100
    visual_tokens = int(seq_len * args.visual_ratio)
    if visual_start_index + visual_tokens + 1 > seq_len:
        visual_tokens = (
            seq_len - visual_start_index - 1
        )  # Adjust if sequence is too short

    # Insert special tokens:
    # Set the vision start token at the beginning of the visual region.
    input_ids[:, visual_start_index] = config.vision_start_token_id
    # Fill the next visual_tokens positions with the image token id.
    input_ids[:, visual_start_index + 1 : visual_start_index + 1 + visual_tokens] = (
        config.image_token_id
    )

    # Create dummy image grid tensor and attention mask.
    image_grid_thw = torch.tensor([[1, 25, 40]])
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=dtype)

    # Get position_ids and any other necessary positional encodings.
    position_ids, mrope_position_deltas = get_rope_index(
        config,
        input_ids,
        image_grid_thw,
        attention_mask=attention_mask,
        video_grid_thw=None,
    )

    # Dummy inputs
    patch_pos = torch.zeros((batch_size, seq_len), device=device, dtype=torch.int8) - 1
    # Mark the visual token region in patch_pos with value 1.
    patch_pos[:, visual_start_index : visual_start_index + visual_tokens] = 1

    # Create select_mask.
    if args.selection:
        select_mask = get_select_mask(patch_pos[0], skip_ratio=0.5).unsqueeze(dim=0)
    else:
        select_mask = torch.ones_like(input_ids, dtype=bool, device=device)

    print("Total num selected tokens:", select_mask.sum().item())

    ## Warm-up runs to avoid initialization overheads.
    warmup_iters = 5
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cache_position=None,
                patch_pos=patch_pos,
                select_mask=select_mask,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    ## Timing multiple iterations.
    num_iters = args.iterations
    times = []
    for i in range(num_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cache_position=None,
                patch_pos=patch_pos,
                select_mask=select_mask,
            )

        end_event.record()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
        print(f"Iteration {i+1}: {elapsed_time:.2f} ms")

    avg_time = sum(times) / len(times)
    print(f"Average model elapsed time over {num_iters} iterations: {avg_time:.2f} ms")


if __name__ == "__main__":
    main()
