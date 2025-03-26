import torch
import time
import torch.nn as nn

# --- Dummy Config and Model Setup ---


# Define a dummy configuration class with minimal required attributes.
class DummyConfig:
    hidden_size = 768
    lm_skip_layer = {0: 1}  # Set to 1 to use ui_guide_forward branch
    lm_skip_ratio = 0.5  # Adjust skip ratio as desired
    use_sliding_window = False  # Disable sliding window for simplicity
    _attn_implementation = "default"  # Replace with appropriate key if needed
    rms_norm_eps = 1e-6


# Dummy attention and MLP classes for testing purposes.
class DummySelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.randn(1))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
    ):
        # For testing, simply return the input as output.
        # Also return a dummy present_key_value if use_cache is True.
        present_key_value = None
        if use_cache:
            present_key_value = (
                torch.zeros_like(hidden_states),
                torch.zeros_like(hidden_states),
            )
        return hidden_states, None, present_key_value


class DummyMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        return self.linear(x)


class DummyRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # Simple normalization for testing
        norm = x.norm(dim=-1, keepdim=True)
        return x / (norm + self.eps)


# Create a mapping for the attention class used in Qwen2VLDecoderLayer.
QWEN2_VL_ATTENTION_CLASSES = {"default": DummySelfAttention}


# Now we define our Qwen2VLDecoderLayer using the dummy components.
class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: DummyConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        lm_skip_layer = getattr(config, "lm_skip_layer", None)
        if lm_skip_layer:
            self.layer_skip = lm_skip_layer[layer_idx]
        else:
            self.layer_skip = 0
        self.layer_skip_ratio = getattr(config, "lm_skip_ratio", 0)
        self.layer_skip_rand = False  # Set False for deterministic behavior in tests

        self.self_attn = QWEN2_VL_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx
        )
        self.mlp = DummyMLP(config)
        self.input_layernorm = DummyRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DummyRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_value: torch.Tensor = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor = None,
        position_embeddings: tuple = None,
        patch_pos: torch.Tensor = None,
        select_mask: torch.Tensor = None,
        **kwargs,
    ):
        if self.layer_skip == 0:
            return self.navie_forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                **kwargs,
            )
        elif self.layer_skip == 1:
            return self.ui_guide_forward(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                patch_pos,
                select_mask,
                **kwargs,
            )
        else:
            raise NotImplementedError

    def navie_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_value: torch.Tensor = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor = None,
        position_embeddings: tuple = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    def ui_guide_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_value: torch.Tensor = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor = None,
        position_embeddings: tuple = None,
        patch_pos: torch.Tensor = None,
        select_mask: torch.Tensor = None,
        **kwargs,
    ):
        device = hidden_states.device
        layer_skip_ratio = getattr(self, "layer_skip_ratio", 0)

        if patch_pos is not None and layer_skip_ratio != 0:
            if select_mask is not None:
                retain_mask = select_mask[0]
            else:
                # Fallback: retain every token (for simplicity)
                retain_mask = torch.ones(
                    hidden_states.size(1), dtype=torch.bool, device=device
                )

            selected_hidden_states = hidden_states[:, retain_mask, :]
            # Adjusting position_ids and cache_position for the retained tokens.
            adjusted_position_ids = (
                position_ids[:, retain_mask] if position_ids is not None else None
            )
            adjusted_cache_position = (
                cache_position[retain_mask] if cache_position is not None else None
            )

            # Adjust position embeddings if provided.
            if position_embeddings is not None:
                cos, sin = position_embeddings
                adjusted_cos = cos[:, retain_mask]
                adjusted_sin = sin[:, retain_mask]
                adjusted_position_embeddings = (adjusted_cos, adjusted_sin)
            else:
                adjusted_position_embeddings = None

            block_outputs = self.navie_forward(
                hidden_states=selected_hidden_states,
                attention_mask=attention_mask,
                position_ids=adjusted_position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=adjusted_cache_position,
                position_embeddings=adjusted_position_embeddings,
                **kwargs,
            )

            processed_hidden_states = hidden_states.clone()
            if use_cache:
                processed_hidden_states[:, retain_mask] = block_outputs[0].flatten(0, 1)
                present_key_value = block_outputs[1]
            else:
                processed_hidden_states[:, retain_mask] = block_outputs[0]
            outputs = (processed_hidden_states,)
            if use_cache:
                outputs += (present_key_value,)
        else:
            outputs = self.navie_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return outputs


# --- Benchmarking the Speed ---

# Instantiate the dummy configuration and model layer.
config = DummyConfig()
layer_idx = 0
layer = Qwen2VLDecoderLayer(config, layer_idx)
layer.eval()  # Set to evaluation mode for benchmarking

# Optionally, move to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer.to(device)

# Define input dimensions.
batch_size = 1
seq_len = 4096

# Create dummy inputs.
hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
attention_mask = torch.ones(batch_size, seq_len, device=device)
position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
cache_position = torch.arange(seq_len, device=device)

# For testing the UI-guided branch, create dummy patch_pos and select_mask.
patch_pos = torch.randint(-1, 10, (batch_size, seq_len), device=device)
# select_mask: here a random boolean mask with shape (1, seq_len)
select_mask = torch.randint(0, 2, (1, seq_len), device=device).bool()

# Warm-up: run a few iterations to ensure any lazy initialization is done.
for _ in range(10):
    _ = layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        patch_pos=patch_pos,
        select_mask=select_mask,
        use_cache=True,
    )

# Timing: run multiple iterations and measure the average time.
iterations = 1000
start_time = time.time()
for _ in range(iterations):
    _ = layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        patch_pos=patch_pos,
        select_mask=select_mask,
        use_cache=True,
    )
    # If using GPU, ensure synchronization after each call.
    if device.type == "cuda":
        torch.cuda.synchronize()
end_time = time.time()

avg_time = (end_time - start_time) / iterations
print(f"Average forward pass time over {iterations} iterations: {avg_time:.6f} seconds")
