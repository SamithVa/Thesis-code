
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class inherited from Qwen2-VL.
"""
import pdb
import torch
from typing import List, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging

from image_processing_qwen2_vl import Qwen2VLImageProcessor
from utils import get_select_mask

logger = logging.get_logger(__name__)


class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class Qwen2VLProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2 tokenizer into a single processor.
    [`~Qwen2VLProcessor.__call__`] and [`~Qwen2VLProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    Returns:
        tuple: 
            - image_grid_thw : Shape `[batch_size, img_height // patch_size // merge_size, img_width // patch_size // merge_size]`
            Example : `[[  1,  78, 138]]`

            - patch_pos : Shape `[batch_size, sequence_length]`.  
            Example: `[[-1, -1, 0, 0, 1, 2, ..., n, n, -1, -1]]`
                - `-1`: Represents a text token.
                - `0 -> n`: Represents UI graph component indices.

            - patch_assign : Shape `[batch_size, visual_seq_len]`, Only visual tokens.  
            Example: `[[0,  1,  2,  3,  4,  5,  6,  6,  6,  6,  6,  7,  8,  9, 10, 10, 10, 10, 10, 10, ..., n]]]`
                - `0 -> n`: Represents UI graph component indices.
            
            - patch_assign_len : Shape `[batch_size]`, Number of total UI Graph Components.
            Example: `[257]` , total of 257 unique components

            - select_mask : Shape `[batch_size, sequence_length]`.  
            Example: `[[True, True, False, ..., True, True]]`
                - `True`: Selected (all text tokens are selected by default).
                - `False`: Not selected.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    # inherited from Qwen2-VL.
    image_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        # inherited from Qwen2-VL.        
        self.image_processor = Qwen2VLImageProcessor(**vars(image_processor))


        # Screenshot -> Graph
        self.uigraph_train = kwargs.get("uigraph_train", True)      # Enable ui graph during training
        self.uigraph_test = kwargs.get("uigraph_test", True)       # Enable ui graph during inference
        self.uigraph_diff = kwargs.get("uigraph_diff", 1)           # Pixel difference used for constructing ui graph
        self.uigraph_rand = kwargs.get("uigraph_rand", False)       # Enable random graph construction 
        # Graph -> Mask
        self.uimask_pre = kwargs.get("uimask_pre", True)           # Prebuild patch selection mask in the preprocessor (not in model layers)
        self.uimask_ratio = kwargs.get("uimask_ratio", 0.5)           # Specify the percentage of patch tokens to skip per component
        self.uimask_rand = kwargs.get("uimask_rand", False)         # Enable random token selection instead of uniform selection

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        vis_dir: str = None,
        training = False,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            vis_dir (`str`, *optional*, defaults to `None`):
                If build, the path to store the image with ui graph visualization.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        # Enable ui graph or not
        if training:
            uigraph_use = self.uigraph_train
        else:
            uigraph_use = self.uigraph_test

        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, 
                                                uigraph_use=uigraph_use, 
                                                uigraph_diff=self.uigraph_diff, 
                                                uigraph_rand=self.uigraph_rand, 
                                                vis_dir=vis_dir,
                                                uimask_ratio=self.uimask_ratio,
                                                **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
            patch_assign_len = image_inputs["patch_assign_len"]
        else:
            image_inputs = {}
            image_grid_thw = None
            patch_assign_len = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token, "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # ui graph
        num_img = len(image_inputs['patch_assign_len'])
        cur_img_idx = 0
        pre_start = 0

        # patch_pos indicates the position of visual patch in the full input seq
        text_inputs['patch_pos'] = torch.zeros_like(text_inputs['input_ids']) -1
        for i in range(len(text_inputs['input_ids'][0])):
            # assume here is 1 x L
            if text_inputs['input_ids'][0, i] == 151652:   # <|vision_start|> in Qwen2VL vocabulary
                cur_img_len = image_inputs['image_grid_thw'][cur_img_idx].prod() // merge_length
                text_inputs['patch_pos'][0, i+1: i+1+cur_img_len] = image_inputs['patch_assign'][pre_start: pre_start+cur_img_len]
                cur_img_idx += 1
                pre_start += cur_img_len
        
        if self.uimask_pre:
            text_inputs['select_mask'] = get_select_mask(text_inputs['patch_pos'][0], 
                                                        skip_ratio=self.uimask_ratio, 
                                                        rand=(training and self.uimask_rand)).unsqueeze(0)
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


# if __name__=="__main__":
#     image_processor = Qwen2VLImageProcessor()
#     image = torch.rand([3, 448, 448])
#     img_emb = image_processor.preprocess([image], videos=None)
#     for key in img_emb.keys():
#         print(f'{key}')