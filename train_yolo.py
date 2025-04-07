import torch
import yaml
import os
import shutil
from ultralytics import YOLO
from PIL import Image
import glob

def create_dataset_yaml(output_path='data/soccer.yaml'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_config = {
        'path': base_dir,
        'train': os.path.join(base_dir, 'RBK_TDT17/1_train-val_1min_aalesund_from_start'),
        'val': os.path.join(base_dir, 'RBK_TDT17/2_train-val_1min_after_goal'),
        'names': {
            0: 'ball',
            1: 'player'
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
            class_id = int(parts[7]) - 1
            
            x, y = float(parts[2]), float(parts[3])
            width, height = float(parts[4]), float(parts[5])
            
            x_center = (x + width / 2) / img_width
            y_center = (y + height / 2) / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            
            if frame_id not in frame_annotations:
                frame_annotations[frame_id] = []
            
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            frame_annotations[frame_id].append(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}")
    
    for old_label in glob.glob(os.path.join(output_dir, '*.txt')):
        os.remove(old_label)
    
    labels_created = 0
    for frame_id, annotations in frame_annotations.items():
        if frame_id in frame_to_image:
            img_filename = frame_to_image[frame_id]
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
        else:
            label_filename = f"{frame_id:06d}.txt"
        
        label_path = os.path.join(output_dir, label_filename)
        with open(label_path, 'w') as f:
            for annotation in annotations:
                f.write(annotation + '\n')
        labels_created += 1
    
    print(f"Converted {labels_created} frames to YOLO format in {output_dir}")
    return labels_created

def train_yolo_model(data_path, weights_path='yolov8s.pt', epochs=50, batch_size=16, img_size=640,
                     save_period=5, device='0', project='runs/soccer', name='player_ball_detection'):
    model = YOLO(weights_path)
    
    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        save_period=save_period,
        device=device,
        project=project,
        name=name,
        verbose=True,
        patience=20,
        cos_lr=True,
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        cache=False
    )
    
    final_weights_path = f"{project}/{name}/weights/best.pt"
    print(f"Training complete. Best model saved to {final_weights_path}")
    
    return model

def modify_image_paths(img_dir, labels_dir):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    first_img = img_files[0] if img_files else None
    
    if first_img and not first_img.isdigit():
        print(f"Renaming images in {img_dir} to match YOLOv8 requirements...")
        for i, img_file in enumerate(img_files, 1):
            new_name = f"{i:06d}.jpg"
            os.rename(
                os.path.join(img_dir, img_file),
                os.path.join(img_dir, new_name)
            )
        print(f"Renamed {len(img_files)} images.")
    
    return True

def prepare_yolo_dataset(check_names=True):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_gt_dir = os.path.join(base_dir, 'RBK_TDT17/1_train-val_1min_aalesund_from_start/gt')
    train_img1_dir = os.path.join(base_dir, 'RBK_TDT17/1_train-val_1min_aalesund_from_start/img1')
    train_images_dir = os.path.join(base_dir, 'RBK_TDT17/1_train-val_1min_aalesund_from_start/images')
    train_labels_dir = os.path.join(base_dir, 'RBK_TDT17/1_train-val_1min_aalesund_from_start/labels')
    
    val_gt_dir = os.path.join(base_dir, 'RBK_TDT17/2_train-val_1min_after_goal/gt')
    val_img1_dir = os.path.join(base_dir, 'RBK_TDT17/2_train-val_1min_after_goal/img1')
    val_images_dir = os.path.join(base_dir, 'RBK_TDT17/2_train-val_1min_after_goal/images')
    val_labels_dir = os.path.join(base_dir, 'RBK_TDT17/2_train-val_1min_after_goal/labels')
    
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    
    print(f"Copying/linking images to YOLOv8 standard directory structure...")
    
    for img_file in os.listdir(train_img1_dir):
        if img_file.endswith('.jpg'):
            src = os.path.join(train_img1_dir, img_file)
            dst = os.path.join(train_images_dir, img_file)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except (OSError, AttributeError):
                    shutil.copy2(src, dst)
    
    for img_file in os.listdir(val_img1_dir):
        if img_file.endswith('.jpg'):
            src = os.path.join(val_img1_dir, img_file)
            dst = os.path.join(val_images_dir, img_file)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except (OSError, AttributeError):
                    shutil.copy2(src, dst)
    
    if check_names:
        modify_image_paths(train_images_dir, train_labels_dir)
        modify_image_paths(val_images_dir, val_labels_dir)
    
    train_count = convert_gt_to_yolo_format(train_gt_dir, train_labels_dir, train_images_dir)
    val_count = convert_gt_to_yolo_format(val_gt_dir, val_labels_dir, val_images_dir)
    
    if train_count == 0 or val_count == 0:
        print("Warning: No labels were created. Please check your ground truth files.")
        return False
    
    train_images_list = set([os.path.splitext(f)[0] for f in os.listdir(train_images_dir) if f.endswith('.jpg')])
    train_labels_list = set([os.path.splitext(f)[0] for f in os.listdir(train_labels_dir) if f.endswith('.txt')])
    
    val_images_list = set([os.path.splitext(f)[0] for f in os.listdir(val_images_dir) if f.endswith('.jpg')])
    val_labels_list = set([os.path.splitext(f)[0] for f in os.listdir(val_labels_dir) if f.endswith('.txt')])
    
    print(f"Training set: {len(train_images_list)} images, {len(train_labels_list)} label files")
    print(f"Validation set: {len(val_images_list)} images, {len(val_labels_list)} label files")
    
    print(f"Dataset prepared successfully with {train_count} training and {val_count} validation frames.")
    return True

def main():
    if not prepare_yolo_dataset():
        print("Error preparing dataset. Exiting.")
        return
    
    data_yaml_path = create_dataset_yaml()
    
    weights_path = 'yolov8s.pt'
    epochs = 50
    batch_size = 8
    img_size = 640
    device = '0' if torch.cuda.is_available() else 'cpu'
    project = 'runs/soccer'
    name = 'player_ball_detection'
    
    print(f"Starting training on device: {device}")
    print(f"Using dataset config: {data_yaml_path}")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    
    model = train_yolo_model(
        data_yaml_path,
        weights_path,
        epochs,
        batch_size,
        img_size,
        save_period=5,
        device=device,
        project=project,
        name=name
    )
    
    print(f"\nTraining complete!")
    print(f"Model weights saved in {project}/{name}/weights/")
    print("Use 'best.pt' for inference on new data.")

if __name__ == '__main__':
    main()