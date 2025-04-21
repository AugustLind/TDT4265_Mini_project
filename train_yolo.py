import torch
import yaml
import os
import shutil
from ultralytics import YOLO
from PIL import Image
import glob

def create_dataset_yaml(output_path='data/soccer.yaml'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    dataset_config = {
        'path': dataset_dir,
        'train': os.path.join(dataset_dir, 'train', 'images'),
        'val': os.path.join(dataset_dir, 'val', 'images'),
        'names': {
            0: 'ball',
            1: 'person'
        },
        'nc': 2
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset configuration saved to {output_path}")
    return output_path

def convert_gt_to_yolo_format(gt_dir, output_dir, img_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    gt_file = os.path.join(gt_dir, 'gt.txt')
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file not found at {gt_file}")
        return 0
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    if not img_files:
        print(f"Error: No JPG images found in {img_dir}")
        return 0
    
    # Read dimensions from first image
    first_img_path = os.path.join(img_dir, img_files[0])
    try:
        with Image.open(first_img_path) as img:
            img_width, img_height = img.size
            print(f"Using image dimensions from {first_img_path}: {img_width}x{img_height}")
    except Exception as e:
        print(f"Error reading image dimensions: {e}")
        print("Using default dimensions of 1920x1080")
        img_width, img_height = 1920, 1080
    
    frame_to_image = {}
    for img_file in img_files:
        try:
            frame_num = int(os.path.splitext(img_file)[0])
            frame_to_image[frame_num] = img_file
        except ValueError:
            continue
    
    frame_annotations = {}
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
                
            frame_id = int(parts[0])
            original_class_id = int(parts[7])
            # remap IDs: 1->ball (0), others->person (1)
            class_id = 0 if original_class_id == 1 else 1
            
            x, y = float(parts[2]), float(parts[3])
            width, height = float(parts[4]), float(parts[5])
            
            # normalize
            x_center = (x + width / 2) / img_width
            y_center = (y + height / 2) / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            frame_annotations.setdefault(frame_id, []).append(
                f"{class_id} {x_center} {y_center} {norm_width} {norm_height}"
            )
    
    # clear old labels in output_dir
    for old_label in glob.glob(os.path.join(output_dir, '*.txt')):
        os.remove(old_label)
    
    labels_created = 0
    for frame_id, annotations in frame_annotations.items():
        # determine label filename
        if frame_id in frame_to_image:
            img_filename = frame_to_image[frame_id]
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
        else:
            label_filename = f"{frame_id:06d}.txt"
        label_path = os.path.join(output_dir, label_filename)
        with open(label_path, 'w') as of:
            of.write("\n".join(annotations) + "\n")
        labels_created += 1
    
    print(f"Converted {labels_created} frames to YOLO format in {output_dir}")
    return labels_created

def train_yolo_model(data_path,
                     weights_path='yolov8s.pt',
                     epochs=50,
                     batch_size=16,
                     img_size=640,
                     save_period=5,
                     device='0',
                     project='runs/soccer',
                     name='player_ball_detection'):
    model = YOLO('yolov8m.pt')
    train_kwargs = {
        'data': data_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'save_period': save_period,
        'device': device,
        'project': project,
        'name': name,
        'cos_lr': True,
        'warmup_epochs': 5.0,
        'patience': 10,
        'mosaic': 1.0,
        'mixup': 0.5,
        'copy_paste': 0.5,
        'hsv_h': 0.02,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'flipud': 0.5,
        'rotate': 15,
        'translate': 0.1,
        'scale': 0.5,
        'lr0': 0.001,
        'lrf': 0.0001,
        'momentum': 0.937,
        'weight_decay': 0.0005,
    }
    results = model.train(**train_kwargs)
    def detect_with_tuned_nms(source, conf_thresh=0.25, iou_thresh=0.3):
        return model.predict(source=source, conf=conf_thresh, iou=iou_thresh, multi_label=True)
    print(f"Training complete. Best model saved to {project}/{name}/weights/best.pt")
    return model, detect_with_tuned_nms

def prepare_yolo_dataset(check_names=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    train_images_dir = os.path.join(dataset_dir, 'train', 'images')
    train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
    val_images_dir = os.path.join(dataset_dir, 'val', 'images')
    val_labels_dir = os.path.join(dataset_dir, 'val', 'labels')

    # ensure dirs
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Define train and val sets
    train_sets = [
        'RBK_TDT17/1_train-val_1min_aalesund_from_start',
        'RBK_TDT17/2_train-val_1min_after_goal'
    ]
    val_set = 'RBK_TDT17/3_test_1min_hamkam_from_start'

    # Copy training images and convert GT
    total_train = 0
    for ts in train_sets:
        img_dir = os.path.join(base_dir, ts, 'img1')
        gt_dir = os.path.join(base_dir, ts, 'gt')
        for img_file in os.listdir(img_dir):
            if img_file.endswith('.jpg'):
                dst = os.path.join(train_images_dir, img_file)
                if not os.path.exists(dst):
                    shutil.copy(os.path.join(img_dir, img_file), dst)
        total_train += convert_gt_to_yolo_format(gt_dir, train_labels_dir, img_dir)

    # Copy validation images and convert GT
    val_img_dir = os.path.join(base_dir, val_set, 'img1')
    val_gt_dir = os.path.join(base_dir, val_set, 'gt')
    for img_file in os.listdir(val_img_dir):
        if img_file.endswith('.jpg'):
            dst = os.path.join(val_images_dir, img_file)
            if not os.path.exists(dst):
                shutil.copy(os.path.join(val_img_dir, img_file), dst)
    total_val = convert_gt_to_yolo_format(val_gt_dir, val_labels_dir, val_img_dir)

    if total_train == 0 or total_val == 0:
        print("Warning: No labels were created. Please check your ground truth files.")
        return False

    # summary
    print(f"Training set: {len(os.listdir(train_images_dir))} images, {len(os.listdir(train_labels_dir))} labels")
    print(f"Validation set: {len(os.listdir(val_images_dir))} images, {len(os.listdir(val_labels_dir))} labels")
    print(f"Dataset prepared: {total_train} train annotations, {total_val} val annotations")
    return True

def main():
    if not prepare_yolo_dataset():
        print("Error preparing dataset. Exiting.")
        return
    data_yaml = create_dataset_yaml()
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Starting training on device: {device}")
    model, _ = train_yolo_model(
        data_path=data_yaml,
        weights_path='yolov8s.pt',
        epochs=50,
        batch_size=8,
        img_size=640,
        save_period=5,
        device=device,
        project='soccer',
        name='player_ball_detection'
    )

if __name__ == '__main__':
    main()
