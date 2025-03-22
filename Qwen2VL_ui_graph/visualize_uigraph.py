import cv2
import np
from skimage.segmentation import mark_boundaries
from PIL import Image
from .utils import get_select_mask

def visualize_uigraph(self, inputs, image_size, patch_size, image):
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

    patch_assign = inputs['patch_assign'][0] # Shape [# visual_patches], e.g [0, 0, 1, 1, ..., N, N] where N is total number of ui components
    
    upscaled_uigraph_assign = np.repeat(np.repeat(patch_assign, patch_size, axis=0), patch_size, axis=1)

    upscaled_uigraph_assign = upscaled_uigraph_assign[:resized_height, :resized_width]

    if isinstance(image, Image.Image):
        image = np.array(image)

    # Assuming grayscale or RGB image
    if image.shape[0] in [1, 3]:  
        image = image.transpose(1, 2, 0)
    elif image.shape[2] in [1, 3]:
        pass
    else:
        raise ValueError("Unexpected image shape: {}".format(image.shape))

    boundaries_image = mark_boundaries(image, upscaled_uigraph_assign, color=(0.4, 1, 0.4))
    boundaries_image = (boundaries_image * 255).astype(np.uint8)
    # Create a mask for white patch dropping
    drop_mask = np.zeros_like(upscaled_uigraph_assign, dtype=bool)
    # Convert to OpenCV format (BGR)
    annotated_image = cv2.cvtColor(boundaries_image, cv2.COLOR_RGB2BGR)

    unique_components = np.unique(patch_assign)
    print(unique_components)

    drop_ratio = 0.5
    for comp_id in unique_components:
    # Get component coordinates
        component_mask = (patch_assign == comp_id)
        y_indices, x_indices = np.where(component_mask)

        if len(y_indices) > 2:  # Only drop if there are more than 2 patches
            num_to_drop = int(len(y_indices) * drop_ratio)

            # Random_drop
            # drop_indices = np.random.choice(len(y_indices), num_to_drop, replace=False)
            
            # Select patches to drop at uniform intervals
            step = max(1, len(y_indices) // num_to_drop) # Ensure even spacing
            drop_indices = np.arange(0, len(y_indices), step)[:num_to_drop] # Select evenly spaced indices
            
            for idx in drop_indices:
                patch_x = x_indices[idx] * patch_size
                patch_y = y_indices[idx] * patch_size
                drop_mask[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size] = True

    # Apply white mask to dropped patches
    image[drop_mask] = 255  # Set Black Color Filled

        # Draw UI graph boundaries
    boundaries_image = mark_boundaries(image, upscaled_uigraph_assign, color=(0.4, 1, 0.4))
    boundaries_image = (boundaries_image * 255).astype(np.uint8)

    # Convert to OpenCV format (BGR)
    annotated_image = cv2.cvtColor(boundaries_image, cv2.COLOR_RGB2BGR)

    for comp_id in unique_components:
        # if comp_id == 0:  # Skip background
        #     continue

        # Find component coordinates
        component_mask = (patch_assign == comp_id)
        y_indices, x_indices = np.where(component_mask)
        # print(y_indices, x_indices)

        if len(y_indices) > 0 and len(x_indices) > 0:
            # Compute centroid of the bounding box
            # print(x_indices, y_indices)
            min_x, max_x = np.min(x_indices), np.max(x_indices)
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            center_x = (min_x + max_x) // 2 * patch_size
            center_y = (min_y + max_y) // 2 * patch_size
            # Draw number on image
            cv2.putText(
                annotated_image, 
                str(comp_id), 
                (center_x + 7, center_y + 14), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.3,  # Font scale
                (255, 0, 0),  # Blue color
                1  # Thickness
            )

    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))