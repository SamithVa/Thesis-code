
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os, ast
from tqdm import tqdm
from PIL import Image
import argparse
import os 
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--imgs_dir', type=str, required=False)
parser.add_argument('--imgs_annot', type=str, required=False, help="json file for reference test data")
# parser.add_argument('--pred_path', type=str, required=True, help="json file for predicted test data")
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
mind2web_train = args.imgs_annot

if mind2web_imgs_dir == None:
    mind2web_imgs_dir = '/data/data1/syc/intern/wanshan/mind2map_dataset/mind2web_images'
if mind2web_train == None:
    mind2web_train = json.load(open('/data/data1/syc/intern/wanshan/mind2map_dataset/mind2web_data_train.json', 'r'))
# Load reference data, true values
ref = []
for episode in tqdm(mind2web_train):
    goal = episode["confirmed_task"]
    annot_id = episode["annotation_id"]
    previous_actions = []
    results_actions = []

    for j, step in enumerate(episode["actions"]):
        if "bbox" not in step:
            print("action not found")
            continue

        filename = annot_id + '-' + step["action_uid"] + '.jpg'
        img_path = os.path.join(mind2web_imgs_dir, filename)
        if not os.path.exists(img_path):
            print("img not found")
            continue
        image = Image.open(img_path)

        previous_step = ""
        for i, action in enumerate(previous_actions[-4:]):
            previous_step += 'Step' + str(i) + ': ' + action + ". "

        action_step = action2step(step, image.size)
        previous_actions.append(action_step)

        # prompt = prompt_origin.format(goal, previous_step)

        action_step_ref, bbox_ref = action2step(step, image.size, return_bbox=True)
        try:
            action_step_ref = ast.literal_eval(action_step_ref)
        except:
            continue
        step_ref = {"annot_id": annot_id, "img_path": img_path,
                       "bbox_ref": bbox_ref, "action_step_ref": action_step_ref}
        results_actions.append(step_ref)
    ref.append(results_actions)

df = [item for sublist in ref for item in sublist]

# Convert into DataFrame format
# df = pd.DataFrame(pred_data)
df = pd.DataFrame(df)

# model responses "sentence" are in string format : parse string -> dict format
# df['sentence'] = df['sentence'].apply(ast.literal_eval)

# merge_df = pd.merge(df, ref_df, on=['annot_id', 'img_path'])
merge_df = df

# op_false_count = merge_df['Op_match'].value_counts().get(False, 0)
# ele_false_count = merge_df['Ele_match'].value_counts().get(False, 0)

# print(f'Operation mismatched count: {op_false_count}/{len(merge_df)} \nElement mismatched count: {ele_false_count}/{len(merge_df)}')

# Save the mismatched data (series of data not individual!)
# op_mismatch_df = merge_df[merge_df['Op_match'] == False]
# op_mismatch_df = merge_df[merge_df['annot_id'].isin(op_mismatch_df['annot_id'])]

# ele_mismatch_df = merge_df[merge_df['Ele_match'] == False]
# ele_mismatch_df = merge_df[merge_df['annot_id'].isin(ele_mismatch_df['annot_id'])]


# mismatch_types = {
#     "operation_mismatch": op_mismatch_df,
#     'element_mismatch': ele_mismatch_df
# }

# Drawing functions
def draw_rectangle(ax, bbox, img_width, img_height):
        """Draws a rectangle on the given axes."""
        x_min = bbox[0] * img_width
        y_min = bbox[1] * img_height
        x_max = bbox[2] * img_width
        y_max = bbox[3] * img_height
        rect_width = x_max - x_min
        rect_height = y_max - y_min

        rect = patches.Rectangle((x_min, y_min), rect_width, rect_height, 
                                linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

def draw_click_point(ax, click_point, img_width, img_height):
    """Draws a click point on the given axes."""
    x = click_point[0] * img_width
    y = click_point[1] * img_height
    ax.scatter(x, y, c='red', s=50, edgecolor='black')

def save_images_with_annotations(imgs_list, bboxes, acts, ref_acts, operations, ref_f1, instructions, output_dir):
    for i, img_path in tqdm(enumerate(imgs_list), total=len(imgs_list), desc="Saving output images"):
        fig, ax = plt.subplots(figsize=(10, 5))
        img = mpimg.imread(img_path)
        img_height, img_width, _ = img.shape
        
        # Plot the image
        ax.imshow(img)

        # Draw rectangle bbox
        draw_rectangle(ax, bboxes[i], img_width, img_height)

        # Draw the click point
        draw_click_point(ax, acts[i]['click_point'], img_width, img_height)

        # Add text at the top of the image
        res_act = acts[i]['action_type']
        ax.text(0, 1.10, f'Response: {operations[res_act]},     {acts[i]},',
                    color='red', transform=ax.transAxes, fontsize=10, ha='left')
        ax.text(0, 1.06, f'Reference: {operations[ref_f1[i][1]]},   {ref_acts[i]},',
                    color='green', transform=ax.transAxes, fontsize=10, ha='left')
        ax.text(0, 1.02, f'Instruction: {instructions[i]}',
                    color='black', transform=ax.transAxes, fontsize=10, ha='left')

        # Hide the axis
        ax.axis('off')

        # Save the image with annotation
        output_path = os.path.join(output_dir, f'image_{i+1}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        # print(f'Saved {output_path}')
        
        # Close the figure to free up memory
        plt.close(fig)

# Saving images 
for index, key in enumerate(merge_df):
    imgs_output_dir = f'{args.output_dir}/{task}_{key}'
    print(f'{key} is processing .....')
    os.makedirs(imgs_output_dir, exist_ok=True)

    filtered_df = merge_df[key]
    imgs_list = filtered_df['img_path']
    op_f1 = filtered_df['Op_F1'].tolist()
    acts = filtered_df['sentence'].tolist()
    ref_acts = filtered_df['action_step_ref'].tolist()
    bboxes = filtered_df['bbox_ref'].tolist()
    instructions = filtered_df['instruction'].tolist()

    # acts_type -> operation
    operations = [' ', ' ' , 'SELECT', 'TYPE','CLICK-HOVER-ENTER']
    save_images_with_annotations(imgs_list, bboxes, acts, ref_acts, operations, op_f1, instructions, imgs_output_dir)
    filtered_df.to_json(f"{imgs_output_dir}/{task}_{key}_annot.json", indent=4)