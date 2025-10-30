import os
import json
import yaml
import pandas as pd
import numpy as np
import ast
import re
import shutil
from shapely.geometry import Polygon, Point, LineString

with open('./deepscores/ds2_dense/ds2_dense/deepscores_train.json') as file:
    trainData = json.load(file)
with open('./deepscores/ds2_dense/ds2_dense/deepscores_test.json') as file:
    testData = json.load(file)

labels = dict()

datasetPath = "deepscores/ds2_dense/ds2_dense/"

train_images = pd.DataFrame(trainData['images'])
train_obboxs = pd.DataFrame(trainData['annotations']).T

test_images = pd.DataFrame(testData['images'])
test_obboxs = pd.DataFrame(testData['annotations']).T

train_dir = './deepscores/ds2_dense/ds2_dense/images/train'
test_dir = './deepscores/ds2_dense/ds2_dense/images/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
image_dir = './deepscores/ds2_dense/ds2_dense/images'

for image_filename in train_images['filename']:
    src_path = os.path.join(image_dir, image_filename)
    dest_path = os.path.join(train_dir, image_filename)
    shutil.move(src_path, dest_path)

for image_filename in test_images['filename']:
    src_path = os.path.join(image_dir, image_filename)
    dest_path = os.path.join(test_dir, image_filename)
    shutil.move(src_path, dest_path)

image_dir = './deepscores/ds2_dense/ds2_dense/images'

raw_labels = pd.read_csv('./deepscores/ds2_dense/ds2_dense/new_labels.csv')
raw_labels['label'] -= 1
unique_labels = raw_labels[['label', 'name']]
unique_labels = unique_labels.drop_duplicates(subset=['label'])
unique_labels = unique_labels.sort_values(by=['label']).reset_index(drop=True)

notebook_dir = os.getcwd()
data_path = os.path.join(notebook_dir, 'deepscores\ds2_dense\ds2_dense')
print("Data directory:", data_path)

def generate_yaml_from_dataframe(df):
    yaml_text = "names:\n"
    for index, row in df.iterrows():
        yaml_text += f"  {row['label']}: {row['name']}\n"
    return yaml_text

def write_yaml_dataset(path, train_path, val_path, label_df=None, filename='deep_scores.yaml'):
    data = {
        'path': path,
        'train': train_path,
    }
    if val_path is not None:
        data['val'] = val_path
    if label_df is not None:
        label_yaml = generate_yaml_from_dataframe(label_df)
        data['names'] = yaml.load(label_yaml, Loader=yaml.SafeLoader)
    yaml_text = "path: {}\ntrain: {}\n".format(path, train_path)
    if val_path is not None:
        yaml_text += "val: {}\n".format(val_path)
    if label_df is not None:
        yaml_text += label_yaml
    with open(filename, 'w') as yaml_file:
        yaml_file.write(yaml_text)

train_path = 'images/train'
val_path = 'images/test'
label_df = unique_labels

write_yaml_dataset(data_path, train_path, val_path, label_df)

train_images.rename(columns={'id': 'img_id'}, inplace=True)
test_images.rename(columns={'id': 'img_id'}, inplace=True)

class_mapping = dict(zip(raw_labels['old_id'].astype(str), raw_labels['label']))

def map_cat_ids_to_classes(cat_ids):
    return [class_mapping.get(str(cat_id)) for cat_id in cat_ids]

def clean_labels(label_list):
    return list({label for label in label_list if label is not None})
    
def select_highest_precedence(label_list):
    return max(label_list)

train_obboxs['label'] = train_obboxs['cat_id'].apply(map_cat_ids_to_classes)
test_obboxs['label'] = test_obboxs['cat_id'].apply(map_cat_ids_to_classes)
train_obboxs['label'] = train_obboxs['label'].apply(clean_labels)
test_obboxs['label'] = test_obboxs['label'].apply(clean_labels)
train_obboxs['label'] = train_obboxs['label'].apply(select_highest_precedence)
test_obboxs['label'] = test_obboxs['label'].apply(select_highest_precedence)

def extract_info(comment):
    duration = re.search(r'duration:(\d+);', comment)
    rel_position = re.search(r'rel_position:(-?\d+);', comment)
    return [int(duration.group(1)) if duration else None, int(rel_position.group(1)) if rel_position else None]
    
train_obboxs[['duration', 'rel_position']] = train_obboxs['comments'].apply(extract_info).tolist()
test_obboxs[['duration', 'rel_position']] = test_obboxs['comments'].apply(extract_info).tolist()

train_obboxs['duration_mask'] = train_obboxs['duration'].notna().astype(int)
test_obboxs['duration_mask'] = test_obboxs['duration'].notna().astype(int)
train_obboxs['duration'] = train_obboxs['duration'].replace(np.nan,-1)
test_obboxs['duration'] = test_obboxs['duration'].replace(np.nan,-1)

train_obboxs['rel_position_mask'] = train_obboxs['rel_position'].notna().astype(int)
test_obboxs['rel_position_mask'] = test_obboxs['rel_position'].notna().astype(int)
train_obboxs['rel_position'] = train_obboxs['rel_position'].replace(np.nan,50)
test_obboxs['rel_position'] = test_obboxs['rel_position'].replace(np.nan,50)

def adjust_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    return [x_min, y_min, x_max, y_max]

train_obboxs['padded_bbox'] = train_obboxs['a_bbox'].apply(adjust_bbox)
test_obboxs['padded_bbox'] = test_obboxs['a_bbox'].apply(adjust_bbox)
train_obboxs['padded_bbox'] = train_obboxs['padded_bbox'].apply(adjust_bbox)
test_obboxs['padded_bbox'] = test_obboxs['padded_bbox'].apply(adjust_bbox)

train_obboxs.reset_index(inplace=True)
test_obboxs.reset_index(inplace=True)
train_obboxs.drop(['cat_id','comments'], axis=1, inplace=True)
test_obboxs.drop(['cat_id','comments'], axis=1, inplace=True)
train_obboxs.rename(columns={'index': 'ann_id'}, inplace=True)
test_obboxs.rename(columns={'index': 'ann_id'}, inplace=True)
train_obboxs['ann_id'] = train_obboxs['ann_id'].astype(int)
test_obboxs['ann_id'] = test_obboxs['ann_id'].astype(int)
train_obboxs['area'] = train_obboxs['area'].astype(int)
test_obboxs['area'] = test_obboxs['area'].astype(int)
train_obboxs['img_id'] = train_obboxs['img_id'].astype(int)
test_obboxs['img_id'] = test_obboxs['img_id'].astype(int)

train_data = pd.merge(train_obboxs, train_images, on='img_id', how='inner')
test_data = pd.merge(test_obboxs, test_images, on='img_id', how='inner')
train_data.drop('ann_ids', axis=1, inplace=True)
test_data.drop('ann_ids', axis=1, inplace=True)

measures_df = pd.read_csv('./deepscores/ds2_dense/ds2_dense/deepscores_train_barlines.csv')
measures_df['label'] -= 1

def convert_str_to_list(coord_str):
    return ast.literal_eval(coord_str)

measures_df['a_bbox'] = measures_df['a_bbox'].apply(convert_str_to_list)
measures_df['o_bbox'] = measures_df['o_bbox'].apply(convert_str_to_list)

measures_df['padded_a_bbox'] = measures_df['padded_a_bbox'].apply(convert_str_to_list)
measures_df['padded_o_bbox'] = measures_df['padded_o_bbox'].apply(convert_str_to_list)

filename_to_dimensions = dict(zip(train_images['filename'], zip(train_images['width'], train_images['height'])))

measures_df['width'] = measures_df['filename'].map(lambda x: filename_to_dimensions.get(x, (np.nan, np.nan))[0])
measures_df['height'] = measures_df['filename'].map(lambda x: filename_to_dimensions.get(x, (np.nan, np.nan))[1])

measures_df_test = pd.read_csv('./deepscores/ds2_dense/ds2_dense/deepscores_test_barlines.csv')
measures_df_test['label'] -= 1

def convert_str_to_list(coord_str):
    return ast.literal_eval(coord_str)

measures_df_test['a_bbox'] = measures_df_test['a_bbox'].apply(convert_str_to_list)
measures_df_test['o_bbox'] = measures_df_test['o_bbox'].apply(convert_str_to_list)

measures_df_test['padded_a_bbox'] = measures_df_test['padded_a_bbox'].apply(convert_str_to_list)
measures_df_test['padded_o_bbox'] = measures_df_test['padded_o_bbox'].apply(convert_str_to_list)

filename_to_dimensions = dict(zip(test_images['filename'], zip(test_images['width'], test_images['height'])))

measures_df_test['width'] = measures_df_test['filename'].map(lambda x: filename_to_dimensions.get(x, (np.nan, np.nan))[0])
measures_df_test['height'] = measures_df_test['filename'].map(lambda x: filename_to_dimensions.get(x, (np.nan, np.nan))[1])

def corners_to_yolo(bbox, img_width, img_height):
    bbox = [max(min(bbox[i], img_width - 1 if i % 2 == 0 else img_height - 1), 0) for i in range(len(bbox))]
    polygon = Polygon([(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)])
    min_rect = polygon.minimum_rotated_rectangle
    if isinstance(min_rect, Point):
        x, y = min_rect.x, min_rect.y
        min_rect = Polygon([(x-1, y-1), (x+1, y-1), (x+1, y+1), (x-1, y+1)])
    elif isinstance(min_rect, LineString):
        x_coords, y_coords = zip(*min_rect.coords)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_rect = Polygon([(min_x-1, min_y-1), (max_x+1, max_y+1), (min_x-1, max_y+1), (max_x+1, min_y-1)])
    corners = np.array(min_rect.exterior.coords[:-1])
    edge1 = np.linalg.norm(corners[1] - corners[0])
    edge2 = np.linalg.norm(corners[2] - corners[1])
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    center = min_rect.centroid
    center_x = center.x / img_width
    center_y = center.y / img_height
    angle = np.rad2deg(np.arctan2(corners[1][1] - corners[0][1], corners[1][0] - corners[0][0]))
    width /= img_width
    height /= img_height
    return [max(0, min(center_x, 1)), max(0, min(center_y, 1)), max(0, min(width, 1)), max(0, min(height, 1))]

def apply_corners_to_yolo(row):
    return corners_to_yolo(row['o_bbox'], row['width'], row['height'])

missing_annotations = measures_df[measures_df['filename'].isin(train_data['filename'])]
train_data = pd.concat([train_data, missing_annotations], ignore_index=True)

train_data['yolo_bbox'] = train_data.apply(apply_corners_to_yolo, axis=1)

missing_annotations = measures_df_test[measures_df_test['filename'].isin(test_data['filename'])]
test_data = pd.concat([test_data, missing_annotations], ignore_index=True)

test_data['yolo_bbox'] = test_data.apply(apply_corners_to_yolo, axis=1)

train_data = train_data[train_data['yolo_bbox']!='invalid']
test_data = test_data[test_data['yolo_bbox']!='invalid']

print(train_data["label"])
train_data = train_data[train_data['label']!=155]
test_data = test_data[test_data['label']!=155]
df_agg = train_data.groupby('filename').agg({
    'yolo_bbox': lambda x: list(x),
    'label': lambda x: list(x)
}).reset_index()

df_test_agg = test_data.groupby('filename').agg({
    'yolo_bbox': lambda x: list(x),
    'label': lambda x: list(x)
}).reset_index()

df_agg['yolo_bbox'] = df_agg['yolo_bbox'].apply(lambda x: list(x) if not isinstance(x, list) else x)

non_list_values = [value for value in df_agg['yolo_bbox'] if (not isinstance(value, list)) or (len(value) == 0)]
print("Non-list values in 'yolo_bbox' column:", non_list_values)

def df_to_yolo_text_format(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for index, row in df.iterrows():
        filename = row['filename']
        yolo_bbox = row['yolo_bbox']
        label = row['label']
        text_file_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
        with open(text_file_path, 'w') as text_file:
            for bbox, class_label in zip(yolo_bbox, label):
                x_center = bbox[0]
                y_center = bbox[1]
                bbox_width = bbox[2]
                bbox_height = bbox[3]
                text_file.write(f"{class_label} {x_center} {y_center} {bbox_width} {bbox_height}\n")

train_label_dir = os.path.join(data_path, "labels", "train")
print(f"train_label_dir:{train_label_dir}")
test_label_dir = os.path.join(data_path,"labels", "test")
print(f"test_label_dir:{test_label_dir}")

df_to_yolo_text_format(df_agg, train_label_dir)
df_to_yolo_text_format(df_test_agg, test_label_dir)