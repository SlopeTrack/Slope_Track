import os
import cv2
import json
from sahi.slicing import slice_coco

print('Creating coco .json files from MOT format')

data_dir = 'slope_track'
output_dir = 'datasets/Detection_Split'

splits = ['train', 'val']  # both splits

# Define categories (adjust if needed)
categories = [{"id": 0, "name": "person"}]

for split in splits:
    split_dir = os.path.join(data_dir, split)
    image_output_dir = os.path.join(output_dir, split, 'images')
    annotation_output_dir = os.path.join(output_dir, split, 'annotations')
    annotation_output_file = os.path.join(annotation_output_dir, f'{split}_annotations.json')

    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(annotation_output_dir, exist_ok=True)

    coco_annotation = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    annotation_id = 1
    image_id = 1

    for seq in os.listdir(split_dir):
        seq_path = os.path.join(split_dir, seq)
        img_dir = os.path.join(seq_path, 'img1')
        gt_file = os.path.join(seq_path, 'gt', 'gt.txt')

        if not os.path.exists(img_dir) or not os.path.exists(gt_file):
            print(f"Skipping {seq}: Missing img1/ or gt.txt")
            continue

        # Load ground truth
        gt_data = {}
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame = int(parts[0])
                track_id = int(parts[1])
                x1 = float(parts[2])
                y1 = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                visibility = float(parts[8]) if len(parts) > 8 else 1.0

                if visibility < 0.1:
                    continue

                bbox = [x1, y1, w, h]
                if frame not in gt_data:
                    gt_data[frame] = []
                gt_data[frame].append({
                    "bbox": bbox,
                    "track_id": track_id
                })

        img_files = sorted(os.listdir(img_dir))
        for i, img_file in enumerate(img_files):
            if not img_file.endswith('.jpg'):
                continue

            frame_idx = int(os.path.splitext(img_file)[0])
            img_path = os.path.join(img_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            height, width = frame.shape[:2]
            image_filename = f"{seq}_frame_{frame_idx:05d}.jpg"
            new_img_path = os.path.join(image_output_dir, image_filename)
            cv2.imwrite(new_img_path, frame)

            coco_annotation["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "height": height,
                "width": width
            })

            annotations = gt_data.get(frame_idx, [])
            for ann in annotations:
                coco_annotation["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": ann["bbox"],
                    "area": ann["bbox"][2] * ann["bbox"][3],
                    "iscrowd": 0
                })
                annotation_id += 1

            image_id += 1

    with open(annotation_output_file, 'w') as f:
        json.dump(coco_annotation, f, indent=4)
    print(f"COCO annotations saved for {split} set at {annotation_output_file}")



print('Slicing images using SAHI')

coco_train_images = os.path.join(output_dir, 'train', 'images')
coco_train_dict, coco_train_path = slice_coco(
    coco_annotation_file_path="datasets/Detection_Split/train/annotations/train_annotations.json",
    image_dir=coco_train_images,
    output_coco_annotation_file_name="sliced_train",
    ignore_negative_samples=False,
    output_dir="datasets/Detection_Split/train1/images",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
    min_area_ratio=0.01,
    #verbose=True
)

coco_val_images = os.path.join(output_dir, 'val', 'images')
coco_dict, coco_path = slice_coco(
    coco_annotation_file_path='datasets/Detection_Split/val/annotations/val_annotations.json',
    image_dir=coco_val_images,
    output_coco_annotation_file_name="sliced_val",
    ignore_negative_samples=False,
    output_dir="datasets/Detection_Split/val1/images",
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
    min_area_ratio=0.01,
    #verbose=True
)
print('Completed: Creating images using SAHI')

print('Converting to YOLO files')

def coco_to_yolo_with_verification(coco_json_path, output_dir, images_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    category_map = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

    # Parse images
    images = {img['id']: img for img in coco_data['images']}
    image_filenames = {img['file_name']: img['id'] for img in coco_data['images']}

    # Verify images exist in the specified image directory
    missing_images = []
    for filename in image_filenames.keys():
        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            missing_images.append(filename)

    if missing_images:
        print("The following images were referenced in the COCO JSON but are missing in the images directory:")
        for img in missing_images:
            print(f" - {img}")
        print("\nContinuing with existing images...\n")


    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        img = images[img_id]


        image_path = os.path.join(images_dir, img['file_name'])
        if not os.path.exists(image_path):
            continue


        img_width = img['width']
        img_height = img['height']


        x_min, y_min, box_width, box_height = ann['bbox']


        x_center = (x_min + box_width / 2) / img_width
        y_center = (y_min + box_height / 2) / img_height
        norm_width = box_width / img_width
        norm_height = box_height / img_height


        category_id = category_map[ann['category_id']]

        # Generate YOLO annotation line
        yolo_annotation = f"{category_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"

        # Write to YOLO .txt file (one file per image)
        txt_file_path = os.path.join(output_dir, f"{os.path.splitext(img['file_name'])[0]}.txt")
        print(img['file_name'], txt_file_path)
        with open(txt_file_path, 'a') as txt_file:
            txt_file.write(yolo_annotation)

    print(f"Conversion complete! YOLO annotations saved in '{output_dir}'.")
    if missing_images:
        print(f"\nNote: {len(missing_images)} images were missing and skipped.")

coco_json_path = "datasets/Detection_Split/train1/images/sliced_train_coco.json"  # Replace with your COCO JSON path
output_dir = "datasets/Detection_Split/train1/labels"  # Replace with your desired YOLO output directory
images_dir = "datasets/Detection_Split/train1/images"  # Directory containing image files

coco_to_yolo_with_verification(coco_json_path, output_dir, images_dir)

coco_json_path = "datasets/Detection_Split/val1/images/sliced_val_coco.json"  # Replace with your COCO JSON path
output_dir = "datasets/Detection_Split/val1/labels"  # Replace with your desired YOLO output directory
images_dir = "datasets/Detection_Split/val1/images"  # Directory containing image files

coco_to_yolo_with_verification(coco_json_path, output_dir, images_dir)

print('Completed: Converting to YOLO files')
