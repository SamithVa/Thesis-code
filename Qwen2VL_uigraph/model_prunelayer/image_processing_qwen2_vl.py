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
"""Image processor class inherited from Qwen2-VL."""

import math, cv2
from typing import Dict, List, Optional, Union

import PIL
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage.segmentation import mark_boundaries

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)

from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    VideoInput,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if (
        isinstance(images, (list, tuple))
        and isinstance(images[0], (list, tuple))
        and is_valid_image(images[0][0])
    ):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


# Copied from transformers.models.llava_next_video.image_processing_llava_next_video.make_batched_videos
def make_batched_videos(videos) -> List[VideoInput]:
    if (
        isinstance(videos, (list, tuple))
        and isinstance(videos[0], (list, tuple))
        and is_valid_image(videos[0][0])
    ):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if isinstance(videos[0], Image.Image):
            return [videos]
        elif len(videos[0].shape) == 4:
            return [list(video) for video in videos]

    elif is_valid_image(videos) and len(videos.shape) == 4:
        return [list(videos)]

    raise ValueError(f"Could not make batched video from {videos}")


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# Implement Union-Find operator for constructing ui patches
class UnionFind:
    def __init__(self, size):
        """
        Initializes a Union-Find (Disjoint Set) data structure.

        :param size: The number of elements in the set.
        - `parent` array keeps track of the parent of each element.
        - Initially, each element is its own parent, forming individual sets.
        """
        self.parent = np.arange(size)

    def find(self, x):
        """
        Finds the representative (root) of the set containing `x`.

        :param x: The element to find the root for.
        :return: The root representative of `x`.
        - Uses path compression to flatten the tree structure,
          making future queries faster by pointing nodes directly to the root.
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """
        Merges the sets containing `x` and `y`.

        :param x: First element.
        :param y: Second element.
        - Uses `find` to determine the root representatives of `x` and `y`.
        - If the roots are different, merges them by setting `y`'s root to `x`'s root.
        """
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parent[py] = px


class Qwen2VLImageProcessor(BaseImageProcessor):
    r"""
    Constructs an image processor that inherited from Qwen2-VL, enable UI-guided visual token selection.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    model_input_names = [
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    ]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.size = {"min_pixels": min_pixels, "max_pixels": max_pixels}
        self.do_convert_rgb = do_convert_rgb

    def rerank_values(self, arr):
        mapping = {}
        new_arr = np.empty_like(arr)
        next_value = 0

        for idx, x in enumerate(arr):
            if x not in mapping:
                mapping[x] = next_value
                next_value += 1
            new_arr[idx] = mapping[x]
        return new_arr

    def _build_uigraph(
        self,
        patches,
        grid_t,
        grid_h,
        grid_w,
        grid_h_half,
        grid_w_half,
        uigraph_threshold,
        channel,
    ):
        num_patches = grid_t * grid_h_half * grid_w_half  # [1, 60, 60]

        uf = UnionFind(num_patches)  # initialize UI graph

        def idx(t, i, j):
            """
            - t * grid_h_half * grid_w_half: current time
            - i * grid_w_half: current row
            - j: column offset in current row
            """
            return t * grid_h_half * grid_w_half + i * grid_w_half + j

        # Compare adjacent patches based on the threshold
        for t in range(grid_t):
            for i in range(grid_h_half):
                for j in range(grid_w_half):
                    current_idx = idx(t, i, j)
                    current_patch = patches[
                        t,
                        i,
                        j,
                        :,
                        :,
                        :,
                        :,
                    ]  # Shape: (channel, temporal_patch_size, patch_size, patch_size)

                    # Compare with right neighbor
                    if j + 1 < grid_w_half:
                        right_patch = patches[
                            t,
                            i,
                            j + 1,
                            :,
                            :,
                            :,
                            :,
                        ]
                        # Compute the difference between the patches
                        diff = np.linalg.norm(current_patch - right_patch)
                        if diff < uigraph_threshold:
                            uf.union(current_idx, idx(t, i, j + 1))

                    # Compare with bottom neighbor
                    if i + 1 < grid_h_half:
                        bottom_patch = patches[
                            t,
                            i + 1,
                            j,
                            :,
                            :,
                            :,
                            :,
                        ]
                        # Compute the difference between the patches
                        diff = np.linalg.norm(current_patch - bottom_patch)
                        if diff < uigraph_threshold:
                            uf.union(current_idx, idx(t, i + 1, j))

        # Flatten and encode the Union-Find assignments
        uigraph_assign_flat = np.array([uf.find(x) for x in range(num_patches)])
        le = LabelEncoder()
        uigraph_assign_flat = le.fit_transform(uigraph_assign_flat)
        uigraph_assign = uigraph_assign_flat.reshape((grid_t, grid_h_half, grid_w_half))
        return uigraph_assign

    def _vis_uigraph(self, uigraph_assign, image_size, patch_size, image):
        """
        Visualize UI Graph with uniform patch dropping.

        Args:
            uigraph_assign: Array mapping each patch to a UI component.
            image_size: Tuple (height, width) of the resized image.
            patch_size: Size of each patch in pixels.
            image: Input image.
            drop_ratio: Proportion of patches to be dropped per component (default: 0.5).

        Returns:
            PIL.Image with modified visualization.
        """
        resized_height, resized_width = image_size[0]

        uigraph_assign = uigraph_assign[
            0
        ]  # Shape [# visual_patches], e.g [0, 0, 1, 1, ..., N, N] where N is total number of ui components

        upscaled_uigraph_assign = np.repeat(
            np.repeat(uigraph_assign, patch_size, axis=0), patch_size, axis=1
        ) # [patches * patch_size, patches * patch_size]

        upscaled_uigraph_assign = upscaled_uigraph_assign[
            :resized_height, :resized_width
        ]

        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        # Assuming grayscale or RGB image
        if image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0) # [c, h, w] -> [h, w, c]
        elif image.shape[2] in [1, 3]:
            pass
        else:
            raise ValueError("Unexpected image shape: {}".format(image.shape))

        boundaries_image = mark_boundaries(
            image, upscaled_uigraph_assign, color=(0.4, 1, 0.4)
        )
        boundaries_image = (boundaries_image * 255).astype(np.uint8)

        return Image.fromarray(boundaries_image)

    def visualize_uigraph(self, patch_assign, image_size, patch_size, image, drop_ratio):
        """
        Visualize UI Graph with uniform patch dropping.

        Args:
            uigraph_assign: Array mapping each patch to a UI component.
            image_size: Tuple (height, width) of the resized image.
            inputs: `Dict` include inputs_ids, attention_mask, patch_assign, ...
            image: Input image.
            drop_ratio: Proportion of patches to be dropped per component (default: 0.5).

        Returns:
            PIL.Image with modified visualization.
        """
        resized_height, resized_width = image_size[0]

        # Shape [# visual_patches], e.g [0, 0, 1, 1, ..., N, N] where N is total number of ui components
        patch_assign = patch_assign[0]

        upscaled_uigraph_assign = np.repeat(
            np.repeat(patch_assign, patch_size, axis=0), patch_size, axis=1
        )

        upscaled_uigraph_assign = upscaled_uigraph_assign[
            :resized_height, :resized_width
        ]

        if isinstance(image, Image.Image):
            image = image.resize((resized_width, resized_height), Image.BILINEAR) # resize to new dimension
            image = np.array(image)

        # Assuming grayscale or RGB image
        if image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)
        elif image.shape[2] in [1, 3]:
            pass
        else:
            raise ValueError("Unexpected image shape: {}".format(image.shape))

        boundaries_image = mark_boundaries(
            image, upscaled_uigraph_assign, color=(0.4, 1, 0.4)
        )
        boundaries_image = (boundaries_image * 255).astype(np.uint8)
        # Create a mask for white patch dropping
        drop_mask = np.zeros_like(upscaled_uigraph_assign, dtype=bool)
        # Convert to OpenCV format (BGR)
        # annotated_image = cv2.cvtColor(boundaries_image, cv2.COLOR_RGB2BGR)

        unique_components = np.unique(patch_assign)

        for comp_id in unique_components:
            # Get component coordinates
            component_mask = patch_assign == comp_id
            y_indices, x_indices = np.where(component_mask)

            if len(y_indices) >= 2 and drop_ratio > 0:  # Only drop if there are more than 2 patches
                num_to_drop = round(len(y_indices) * drop_ratio)
                if num_to_drop > 0:
                    # Random_drop
                    # drop_indices = np.random.choice(len(y_indices), num_to_drop, replace=False)

                    # Select patches to drop at uniform intervals
                    step = max(1, len(y_indices) // num_to_drop)  # Ensure even spacing
                    drop_indices = np.arange(0, len(y_indices), step)[
                        :num_to_drop
                    ]  # Select evenly spaced indices

                    for idx in drop_indices:
                        patch_x = x_indices[idx] * patch_size
                        patch_y = y_indices[idx] * patch_size
                        drop_mask[
                            patch_y : patch_y + patch_size, patch_x : patch_x + patch_size
                        ] = True

        # Apply white mask to dropped patches
        image[drop_mask] = 128  # Set Gray Color Filled


        # Convert to OpenCV format (BGR)
        annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # put patch assign id in every patch
        # for comp_id in unique_components:
        #     # if comp_id == 0:  # Skip background
        #     #     continue

        #     # Find component coordinates
        #     component_mask = patch_assign == comp_id
        #     y_indices, x_indices = np.where(component_mask)
        #     # print(y_indices, x_indices)

        #     if len(y_indices) > 0 and len(x_indices) > 0:
        #         # Compute centroid of the bounding box
        #         # print(x_indices, y_indices)
        #         min_x, max_x = np.min(x_indices), np.max(x_indices)
        #         min_y, max_y = np.min(y_indices), np.max(y_indices)
        #         center_x = (min_x + max_x) // 2 * patch_size
        #         center_y = (min_y + max_y) // 2 * patch_size
        #         # Draw number on image
        #         cv2.putText(
        #             annotated_image,
        #             str(comp_id),
        #             (center_x + 7, center_y + 14),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.3,  # Font scale
        #             (255, 0, 0),  # Blue color
        #             1,  # Thickness
        #         )

        # Convert back to PIL Image
        return Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        uigraph_use: bool = False,
        uigraph_diff: float = 0.0,
        uigraph_rand: bool = False,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`List[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            uigraph_use (`bool`, *optional*, defaults to `False`):
                Whether to build ui graph.
            uigraph_diff (`float`, *optional*, defaults to `0.0`):
                If build, this parameter sets the patch-wise difference threshold.
                A larger threshold results in sparser components, while a smaller threshold leads to denser components.
            uigraph_rand (`bool`, *optional*, defaults to `False`):
                If build, whether to build it randomly for ablation studies.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        processed_resize = []  # for visualization
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                image = resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    input_data_format=input_data_format,
                )

            if do_rescale:
                image = self.rescale(
                    image, scale=rescale_factor, input_data_format=input_data_format
                )

            if do_normalize:
                image = self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )

            image = to_channel_dimension_format(
                image, data_format, input_channel_dim=input_data_format
            )
            processed_images.append(image)
            processed_resize.append((resized_height, resized_width))

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = (
            resized_height // self.patch_size,
            resized_width // self.patch_size,
        )

        # default grid as init. ui graph
        grid_h_half = grid_h // self.merge_size
        grid_w_half = grid_w // self.merge_size
        uigraph_assign = np.arange(grid_t * grid_h_half * grid_w_half).reshape(
            (grid_t, grid_h_half, grid_w_half)
        )

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

        # use ui graph construction
        if uigraph_use:
            uigraph_assign = self._build_uigraph(
                patches=patches,
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
                grid_h_half=grid_h_half,
                grid_w_half=grid_w_half,
                uigraph_threshold=uigraph_diff,
                channel=channel,
            )  # flatten patches,  [0, 1, 1, 2, ..., n, n] |  shape [# patches]

        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        # print(uigraph_assign.shape)

        return (
            flatten_patches,
            (grid_t, grid_h, grid_w),
            uigraph_assign,
            processed_resize,
        )

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        uigraph_use: bool = False,
        uigraph_diff: float = 0.0,
        uigraph_rand: bool = False,
        uimask_ratio: float = 0.0,
        vis_dir: str = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            videos (`VideoInput`):
                Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
                passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            uigraph_use (`bool`, *optional*, defaults to `False`):
                Whether to build ui graph.
            uigraph_diff (`float`, *optional*, defaults to `0.0`):
                If build, this parameter sets the patch-wise difference threshold.
                A larger threshold results in sparser components, while a smaller threshold leads to denser components.
            uigraph_rand (`bool`, *optional*, defaults to `False`):
                If build, whether to build it randomly for ablation studies.
            vis_dir (`str`, *optional*, defaults to `None`):
                If build, the path to store the image with ui graph visualization.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = (
            do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        )

        if images is not None:
            images = make_batched_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if images is not None:
            pixel_values, vision_grid_thws = [], []

            patch_assign_sep = (
                []
            )  # store the patch-wise assignment separately for each ui graph
            patch_assign_len = []  # store the component number per ui graph
            patch_assign_shared = (
                []
            )  # store the patch-wise assignment jointly with shared component idx

            for image in images:
                patches, image_grid_thw, uigraph_assign, image_resize = (
                    self._preprocess(
                        image,
                        do_resize=do_resize,
                        resample=resample,
                        do_rescale=do_rescale,
                        rescale_factor=rescale_factor,
                        do_normalize=do_normalize,
                        image_mean=image_mean,
                        image_std=image_std,
                        data_format=data_format,
                        do_convert_rgb=do_convert_rgb,
                        input_data_format=input_data_format,
                        uigraph_use=uigraph_use,
                        uigraph_diff=uigraph_diff,
                        uigraph_rand=uigraph_rand,
                    )
                )

                # if use uigraph
                if uigraph_use:
                    # if apply uigraph_rand
                    if uigraph_rand:
                        C = len(np.unique(uigraph_assign))
                        _, H, W = uigraph_assign.shape
                        uigraph_assign = np.random.randint(0, C + 1, size=(1, H, W))

                # flat 2d graph to 1d
                uigraph_assign_1d = uigraph_assign.flatten()
                uigraph_assign_1d = self.rerank_values(uigraph_assign_1d)
                uigraph_assign_len = len(np.unique(uigraph_assign_1d))

                uigraph_assign_1d += sum(
                    patch_assign_len
                )  # shared component idx to distinguish different images
                patch_assign_shared.extend(uigraph_assign_1d)
                patch_assign_sep.extend(uigraph_assign_1d)
                patch_assign_len.append(uigraph_assign_len)

                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)

                if vis_dir is not None:
                    # image_vis = self._vis_uigraph(
                    #     uigraph_assign,
                    #     image_resize,
                    #     self.patch_size * self.merge_size,
                    #     image,
                    # )
                    
                    image_vis = self.visualize_uigraph(
                        uigraph_assign,
                        image_resize,
                        self.patch_size * self.merge_size,
                        image,
                        drop_ratio = uimask_ratio
                    )

                    # pre_num = np.prod(uigraph_assign.shape).item()
                    # post_num = len(np.unique(uigraph_assign))
                    # img_size = f'{image_resize[0][0]}x{image_resize[0][1]}'
                    # image_vis.save(f'{vis_dir}/{img_size}_{pre_num}_{post_num}.png')
                    image_vis.save(f"{vis_dir}/demo.png")

            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            patch_assign_shared = np.array(patch_assign_shared)

            data = {
                "pixel_values": pixel_values,
                "image_grid_thw": vision_grid_thws,
                "patch_assign": patch_assign_shared,
                "patch_assign_sep": patch_assign_sep,
                "patch_assign_len": patch_assign_len,
            }

        if videos is not None:
            pixel_values, vision_grid_thws = [], []
            for images in videos:
                # uigraph not support video yet
                patches, video_grid_thw, _, _ = self._preprocess(
                    images,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(video_grid_thw)
            pixel_values = np.array(pixel_values)
            vision_grid_thws = np.array(vision_grid_thws)
            data = {
                "pixel_values_videos": pixel_values,
                "video_grid_thw": vision_grid_thws,
            }

        return BatchFeature(data=data, tensor_type=return_tensors)
