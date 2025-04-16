import json
import pandas as pd
import matplotlib.patches as patches
import os, ast
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import argparse
import os 
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--imgs_dir', type=str, required=False)
parser.add_argument('--imgs_annot', type=str, required=False, help="json file for reference test data")
parser.add_argument('--pred_path', type=str, required=True, help="json file for predicted test data")
parser.add_argument('--output_dir', type=str, required=True, help="output directory for images")
args = parser.parse_args()


# convert actions -> step action format
def action2step(action, image_size, return_bbox=False):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{item:.2f}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])

    if return_bbox:
        bbox = [action["bbox"]["x"], action["bbox"]["y"], action["bbox"]["x"] + action["bbox"]["width"],
                action["bbox"]["y"] + action["bbox"]["height"]]
        bbox = [bbox[0] / image_size[0], bbox[1] / image_size[1], bbox[2] / image_size[0], bbox[3] / image_size[1]]
        bbox = [round(item, 3) for item in bbox]

    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(2, click_point,
                                                                                               select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(3, click_point,
                                                                                               typed_text)

    if return_bbox:
        return action_step, bbox
    else:
        return action_step

task = args.task
mind2web_imgs_dir = args.imgs_dir


# Load the predicted result dataset (model responses) from a JSON file
pred_path = args.pred_path
pred_data = json.load(open(pred_path, 'r'))

# Unload data from flattened format
pred_data = [item for sublist in pred_data for item in sublist]

# Convert into DataFrame format
df = pd.DataFrame(pred_data)
# drop two column because they are the same as ref 



# model responses "sentence" are in string format : parse string -> dict format
df['sentence'] = df['sentence'].apply(lambda x: ast.literal_eval(x[0]))


op_false_count = df['Op_match'].value_counts().get(False, 0)
ele_false_count = df['Ele_match'].value_counts().get(False, 0)

print(f'Operation mismatched count: {op_false_count}/{len(df)} \nElement mismatched count: {ele_false_count}/{len(df)}')

# Save the mismatched data (series of data not individual!)
op_mismatch_df = df[df['Op_match'] == False]
op_mismatch_df = df[df['annot_id'].isin(op_mismatch_df['annot_id'])]

ele_mismatch_df = df[df['Ele_match'] == False]
ele_mismatch_df = df[df['annot_id'].isin(ele_mismatch_df['annot_id'])]

mismatch_types = {
    # "operation_mismatch": op_mismatch_df,
    'element_mismatch': ele_mismatch_df
}

# def draw_rectangle(ax, bbox, img_width, img_height):
#     """
#     Example function stub for drawing a bounding box.
#     'bbox' might be of form [x, y, w, h] in pixel coordinates.
#     """
#     if bbox:
#         rect = patches.Rectangle(
#             (bbox[0], bbox[1]),
#             bbox[2],
#             bbox[3],
#             linewidth=2,
#             edgecolor='blue',
#             facecolor='none'
#         )
#         ax.add_patch(rect)

# def draw_click_point(ax, click_point, img_width, img_height):
#     """
#     Example function stub for drawing a click point.
#     'click_point' might be of form (x, y) in pixel coordinates.
#     """
#     if click_point:
#         ax.plot(
#             click_point[0],
#             click_point[1],
#             marker='o',
#             markersize=6,
#             markeredgecolor='yellow',
#             markerfacecolor='red'
#         )

OPERATIONS = [' ', ' ' , 'SELECT', 'TYPE','CLICK']
# OPERATIONS = [' ', ' ' , 'SELECT', 'TYPE','CLICK-HOVER-ENTER']


def load_font(font_size):
    """
    Tries to load a truetype font; falls back to default if unavailable.
    """
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            print("Warning: TrueType font not found. Using default font (this may appear small).")
            return ImageFont.load_default(font_size)

def save_images_with_annotations(
    images, bboxes, pred_action, ref_action, instructions, output_dir, annot_ids
):
    """
    Creates a new image that retains the original resolution of the image content but adds extra padding at the top.
    Only the bounding box and click point (provided as relative positions) are drawn on the original image.
    
    Parameters:
        images       : List of file paths to the images.
        bboxes       : List of bounding boxes for each image as relative coordinates [x, y, w, h] (values between 0 and 1).
        pred_action  : List of dictionaries that include a "click_point" key with a relative (x, y) tuple.
        ref_action   : Ignored in this version.
        instructions : Ignored in this version.
        output_dir   : Directory to save the annotated images.
        annot_ids    : List of IDs for naming the output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    padding_top = 120  # Extra top padding in pixels
    font_size = 20     # Larger text size for readability
    font = load_font(font_size)

    last_id = None
    count = 0

    for i, img_path in tqdm(enumerate(images), total=len(images), desc="Saving images"):
        # Open the original image and get its dimensions
        with Image.open(img_path) as im:
            img_width, img_height = im.size

            # Create a new image with additional top padding
            new_img_height = img_height + padding_top
            new_im = Image.new("RGB", (img_width, new_img_height), color=(255, 255, 255))
            
            # Paste the original image starting at (0, padding_top)
            new_im.paste(im, (0, padding_top))
            
            draw = ImageDraw.Draw(new_im)
            
            # Convert relative bounding box coordinates to absolute pixel positions
            if bboxes[i]:
                # Assume bbox is given as [x, y, w, h] in relative values
                rel_bbox = bboxes[i]
                abs_x = rel_bbox[0] * img_width
                abs_y = rel_bbox[1] * img_height  # Original image coordinate
                abs_x_high = rel_bbox[2] * img_width
                abs_y_high = rel_bbox[3] * img_height
                # Add offset for top padding
                bbox_coords = [abs_x, abs_y + padding_top, abs_x_high, abs_y_high + padding_top]
                draw.rectangle(bbox_coords, outline="green", width=3)

            # Convert relative click point to absolute pixel values and apply the vertical offset
            pred_cp = pred_action[i].get("click_point")
            if pred_cp:
                abs_cx = pred_cp[0] * img_width
                abs_cy = pred_cp[1] * img_height + padding_top
                r = 5  # Radius for the click point marker
                draw.ellipse([abs_cx - r, abs_cy - r, abs_cx + r, abs_cy + r], fill="red", outline="yellow", width=1)

            # Draw the text in the extra padded area at the top
            text_start_y = 10   # Start drawing text 10 pixels from the top edge of the canvas
            line_spacing = font_size + 10  # Use font size plus extra spacing for each new line

            # index to corresponding action type 
            pred_action_str = OPERATIONS[pred_action[i].get('action_type')]
            ref_action_str = OPERATIONS[ref_action[i].get('action_type')]

            draw.text((10, text_start_y), f"Response: {pred_action_str}, click_point: {pred_cp}", fill="green", font=font)
            draw.text((10, text_start_y + line_spacing), f"Reference: {ref_action_str}, click_point: {ref_action[i].get('click_point')}", fill="red", font=font)
            draw.text((10, text_start_y + 2 * line_spacing), f"Instruction: {instructions[i]}", fill="black", font=font)

            # Manage file naming to avoid overwrites
            if annot_ids[i] != last_id:
                count = 1
            else:
                count += 1
            last_id = annot_ids[i]

            out_path = os.path.join(output_dir, f"{annot_ids[i]}_{count}.png")
            new_im.save(out_path)

# Saving images 
for index, key in enumerate(mismatch_types):
    imgs_output_dir = f'{args.output_dir}/{task}_{key}'
    print(f'{key} is processing .....')
    os.makedirs(imgs_output_dir, exist_ok=True)

    filtered_df = mismatch_types[key]
    # print(filtered_df['annot_id'])
    annot_ids = filtered_df['annot_id'].tolist()
    imgs_list = filtered_df['img_path']
    op_f1 = filtered_df['Op_F1'].tolist()
    acts = filtered_df['sentence'].tolist()
    ref_acts = filtered_df['action_step_ref'].tolist()
    bboxes = filtered_df['bbox_ref'].tolist()
    instructions = filtered_df['instruction'].tolist()

    # print(annot_ids)
    # acts_type -> operation
    
    save_images_with_annotations(imgs_list, bboxes, acts, ref_acts, instructions, imgs_output_dir, annot_ids)
    # filtered_df.to_json(f"{imgs_output_dir}/{task}_{key}_annot.json", indent=4)