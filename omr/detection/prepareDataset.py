import json
import yaml
import pandas as pd
import numpy as np
import ast
import re
import shutil
from pathlib import Path
from shapely.geometry import Polygon, Point, LineString

base_path = Path(__file__).resolve().parent / 'datasets'
source_dataset_path = base_path / 'deepscores' / 'ds2_dense'
train_json_path = source_dataset_path / 'deepscores_train.json'
test_json_path = source_dataset_path / 'deepscores_test.json'

with open(train_json_path) as file:
    trainData = json.load(file)
with open(test_json_path) as file:
    testData = json.load(file)

train_images = pd.DataFrame(trainData['images'])
train_obboxs = pd.DataFrame(trainData['annotations']).T
test_images = pd.DataFrame(testData['images'])
test_obboxs = pd.DataFrame(testData['annotations']).T

dataset_path = base_path / 'deepscores' / 'ds2_dense_prepared'
dataset_path.mkdir(parents=True, exist_ok=True)

train_dir_relative = "images/train"
test_dir_relative = "images/test"
train_dir = dataset_path / train_dir_relative
test_dir = dataset_path / test_dir_relative
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

print("Folder generation complete.")

for image_filename in train_images['filename']:
    src_path = source_dataset_path / 'images' / image_filename
    dest_path = train_dir / Path(image_filename).name
    if src_path.exists():
        shutil.copy(str(src_path), str(dest_path))
for image_filename in test_images['filename']:
    src_path = source_dataset_path / 'images' / image_filename
    dest_path = test_dir / Path(image_filename).name
    if src_path.exists():
        shutil.copy(str(src_path), 
        str(dest_path))

print("Image copying complete.")

labels = pd.read_csv(source_dataset_path / "new_labels.csv")
labels['label'] -= 1

# If using all available labels, you need to preprocess them (as they come from two datasets and overlap)
# unique_labels = labels[['label', 'name']].drop_duplicates(subset=['label']).sort_values(by=['label']).reset_index(drop=True)

def write_yaml_dataset(path, train_path, val_path, label_df=None, filename='deep_scores.yaml'):  
    label_yaml = "names:\n"
    for index, row in label_df.iterrows():
        label_yaml += f"  {row['label']}: {row['name']}\n"

    yaml_text = f"path: {path}\ntrain: {train_path}\nval: {val_path}\n" + label_yaml

    with open(filename, 'w') as yaml_file:
        yaml_file.write(yaml_text)

write_yaml_dataset(dataset_path, train_dir_relative, test_dir_relative, labels, filename=dataset_path / 'deep_scores.yaml')

print("YAML created.")

train_images.rename(columns={'id': 'img_id'}, inplace=True)
test_images.rename(columns={'id': 'img_id'}, inplace=True)
class_mapping = dict(zip(labels['old_id'].astype(str), labels['label']))

def map_cat_ids_to_classes(cat_ids):
    return [class_mapping.get(str(cat_id)) for cat_id in cat_ids]
def clean_labels(label_list):
    return list({label for label in label_list if label is not None})
def select_highest_precedence(label_list):
    return max(label_list)
def extract_info(comment):
    duration = re.search(r'duration:(\d+);', comment)
    rel_position = re.search(r'rel_position:(-?\d+);', comment)
    return [int(duration.group(1)) if duration else None, int(rel_position.group(1)) if rel_position else None]

train_obboxs['label'] = train_obboxs['cat_id'].apply(map_cat_ids_to_classes).apply(clean_labels).apply(select_highest_precedence)
test_obboxs['label'] = test_obboxs['cat_id'].apply(map_cat_ids_to_classes).apply(clean_labels).apply(select_highest_precedence)
train_obboxs[['duration', 'rel_position']] = train_obboxs['comments'].apply(extract_info).tolist()
test_obboxs[['duration', 'rel_position']] = test_obboxs['comments'].apply(extract_info).tolist()

for df in [train_obboxs, test_obboxs]:
    df['duration_mask'] = df['duration'].notna().astype(int)
    df['rel_position_mask'] = df['rel_position'].notna().astype(int)
    df['duration'] = df['duration'].replace(np.nan, -1)
    df['rel_position'] = df['rel_position'].replace(np.nan, 50)

def adjust_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    return [x_min, y_min, x_max, y_max]

for df in [train_obboxs, test_obboxs]:
    df['padded_bbox'] = df['a_bbox'].apply(adjust_bbox).apply(adjust_bbox)

train_obboxs.reset_index(inplace=True)
test_obboxs.reset_index(inplace=True)
train_obboxs.drop(['cat_id','comments'], axis=1, inplace=True)
test_obboxs.drop(['cat_id','comments'], axis=1, inplace=True)
train_obboxs.rename(columns={'index': 'ann_id'}, inplace=True)
test_obboxs.rename(columns={'index': 'ann_id'}, inplace=True)
for df in [train_obboxs, test_obboxs]:
    df['ann_id'] = df['ann_id'].astype(int)
    df['area'] = df['area'].astype(int)
    df['img_id'] = df['img_id'].astype(int)

train_data = pd.merge(train_obboxs, train_images, on='img_id', how='inner').drop('ann_ids', axis=1)
test_data = pd.merge(test_obboxs, test_images, on='img_id', how='inner').drop('ann_ids', axis=1)

measures_df = pd.read_csv(source_dataset_path / 'deepscores_train_barlines.csv')
measures_df['label'] -= 1

def convert_str_to_list(coord_str): return ast.literal_eval(coord_str)
for col in ['a_bbox', 'o_bbox', 'padded_a_bbox', 'padded_o_bbox']:
    measures_df[col] = measures_df[col].apply(convert_str_to_list)

filename_to_dimensions = dict(zip(train_images['filename'], zip(train_images['width'], train_images['height'])))
measures_df['width'] = measures_df['filename'].map(lambda x: filename_to_dimensions.get(x, (np.nan, np.nan))[0])
measures_df['height'] = measures_df['filename'].map(lambda x: filename_to_dimensions.get(x, (np.nan, np.nan))[1])

measures_df_test = pd.read_csv(source_dataset_path / 'deepscores_test_barlines.csv')
measures_df_test['label'] -= 1

for col in ['a_bbox', 'o_bbox', 'padded_a_bbox', 'padded_o_bbox']:
    measures_df_test[col] = measures_df_test[col].apply(convert_str_to_list)

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
        min_rect = Polygon([(min_x-1, min_y-1), (max_x+1, min_y+1), (min_x-1, max_y+1), (max_x+1, min_y-1)])
    corners = np.array(min_rect.exterior.coords[:-1])
    edge1 = np.linalg.norm(corners[1] - corners[0])
    edge2 = np.linalg.norm(corners[2] - corners[1])
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    center = min_rect.centroid
    center_x = center.x / img_width
    center_y = center.y / img_height
    width /= img_width
    height /= img_height
    return [max(0, min(center_x, 1)), max(0, min(center_y, 1)), max(0, min(width, 1)), max(0, min(height, 1))]

def apply_corners_to_yolo(row): return corners_to_yolo(row['o_bbox'], row['width'], row['height'])

missing_annotations = measures_df[measures_df['filename'].isin(train_data['filename'])]
train_data = pd.concat([train_data, missing_annotations], ignore_index=True)
train_data['yolo_bbox'] = train_data.apply(apply_corners_to_yolo, axis=1)
missing_annotations = measures_df_test[measures_df_test['filename'].isin(test_data['filename'])]
test_data = pd.concat([test_data, missing_annotations], ignore_index=True)
test_data['yolo_bbox'] = test_data.apply(apply_corners_to_yolo, axis=1)
train_data = train_data[train_data['yolo_bbox']!='invalid']
test_data = test_data[test_data['yolo_bbox']!='invalid']
train_data = train_data[train_data['label']!=155]
test_data = test_data[test_data['label']!=155]

df_agg = train_data.groupby('filename').agg({'yolo_bbox': lambda x: list(x), 'label': lambda x: list(x)}).reset_index()
df_test_agg = test_data.groupby('filename').agg({'yolo_bbox': lambda x: list(x), 'label': lambda x: list(x)}).reset_index()
df_agg['yolo_bbox'] = df_agg['yolo_bbox'].apply(lambda x: list(x) if not isinstance(x, list) else x)

def df_to_yolo_text_format(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        filename = row['filename']
        yolo_bbox = row['yolo_bbox']
        label = row['label']
        text_file_path = output_dir / (Path(filename).stem + '.txt')
        with open(text_file_path, 'w') as text_file:
            for bbox, class_label in zip(yolo_bbox, label):
                text_file.write(f"{class_label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

train_label_dir = dataset_path / 'labels' / 'train'
test_label_dir = dataset_path / 'labels' / 'test'
df_to_yolo_text_format(df_agg, train_label_dir)
df_to_yolo_text_format(df_test_agg, test_label_dir)

print("Done.")
