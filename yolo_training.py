import os
import yaml
from ultralytics import YOLO
import torch
import shutil

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset') # Standard YOLO dataset location
RAW_DATA_DIR = os.path.join(BASE_DIR, 'RBK_TDT17')

# Define dataset sources
TRAIN_SRC_DIRS = [
    os.path.join(RAW_DATA_DIR, '1_train-val_1min_aalesund_from_start'),
    os.path.join(RAW_DATA_DIR, '2_train-val_1min_after_goal')
]
# Note: Using 2_... also for validation during training as per standard practice.
VAL_SRC_DIR = os.path.join(RAW_DATA_DIR, '2_train-val_1min_after_goal')
TEST_SRC_DIR = os.path.join(RAW_DATA_DIR, '3_test_1min_hamkam_from_start')
FINAL_VAL_SRC_DIR = os.path.join(RAW_DATA_DIR, '4_annotate_1min_bodo_start')

# Define target directories within DATASET_DIR
TRAIN_IMG_DST = os.path.join(DATASET_DIR, 'train', 'images')
TRAIN_LBL_DST = os.path.join(DATASET_DIR, 'train', 'labels')
VAL_IMG_DST = os.path.join(DATASET_DIR, 'val', 'images')
VAL_LBL_DST = os.path.join(DATASET_DIR, 'val', 'labels')
TEST_IMG_DST = os.path.join(DATASET_DIR, 'test', 'images')
TEST_LBL_DST = os.path.join(DATASET_DIR, 'test', 'labels')
FINAL_VAL_IMG_DST = os.path.join(DATASET_DIR, 'final_val', 'images')
FINAL_VAL_LBL_DST = os.path.join(DATASET_DIR, 'final_val', 'labels')

DATA_YAML_PATH = os.path.join(BASE_DIR, 'soccer_data.yaml')

# Training parameters
MODEL_NAME = 'yolov8m.pt' # Pretrained model to start from
BATCH_SIZE = 16
IMG_SIZE = 1024
PROJECT_NAME = 'runs/soccer_training'
RUN_NAME = 'exp'
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# --- Helper Functions ---

def copy_files(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    """Copies images and labels from source to destination."""
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    # Check if source directories exist
    if not os.path.isdir(src_img_dir):
        print(f"Warning: Source image directory not found: {src_img_dir}")
        return
    if not os.path.isdir(src_lbl_dir):
        print(f"Warning: Source label directory not found: {src_lbl_dir}")
        # Attempt to continue with images only if labels are missing
        # return # Uncomment this line if labels are strictly required

    print(f"Copying images from {src_img_dir} to {dst_img_dir}")
    for item in os.listdir(src_img_dir):
        s = os.path.join(src_img_dir, item)
        d = os.path.join(dst_img_dir, item)
        if os.path.isfile(s) and item.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(s, d)

    if os.path.isdir(src_lbl_dir):
        print(f"Copying labels from {src_lbl_dir} to {dst_lbl_dir}")
        for item in os.listdir(src_lbl_dir):
            s = os.path.join(src_lbl_dir, item)
            d = os.path.join(dst_lbl_dir, item)
            if os.path.isfile(s) and item.lower().endswith('.txt'):
                shutil.copy2(s, d)
    else:
         print(f"Skipping label copy as source directory doesn't exist: {src_lbl_dir}")


def prepare_yolo_dataset():
    """
    Prepares the dataset structure required by YOLO.
    Assumes source data has 'img1' and 'labels' subdirectories.
    If your labels are in a 'gt' directory or need conversion (like from MOT format),
    you MUST adapt this function or run a separate conversion script first.
    """
    print("Preparing dataset structure...")

    # --- Training Data ---
    # Clear existing training data first
    if os.path.exists(TRAIN_IMG_DST): shutil.rmtree(TRAIN_IMG_DST)
    if os.path.exists(TRAIN_LBL_DST): shutil.rmtree(TRAIN_LBL_DST)
    # Copy from all specified training sources
    for src_dir in TRAIN_SRC_DIRS:
        src_img = os.path.join(src_dir, 'img1')
        src_lbl = os.path.join(src_dir, 'labels') # ASSUMES labels are already in YOLO format in 'labels' folder
        if not os.path.exists(src_lbl):
             src_lbl = os.path.join(src_dir, 'gt') # Fallback check for 'gt' folder
             print(f"Warning: Using 'gt' folder for labels from {src_dir}. Ensure these are in YOLO format.")
        copy_files(src_img, src_lbl, TRAIN_IMG_DST, TRAIN_LBL_DST)

    # --- Validation Data (during training) ---
    if os.path.exists(VAL_IMG_DST): shutil.rmtree(VAL_IMG_DST)
    if os.path.exists(VAL_LBL_DST): shutil.rmtree(VAL_LBL_DST)
    val_src_img = os.path.join(VAL_SRC_DIR, 'img1')
    val_src_lbl = os.path.join(VAL_SRC_DIR, 'labels') # ASSUMES labels are already in YOLO format in 'labels' folder
    if not os.path.exists(val_src_lbl):
        val_src_lbl = os.path.join(VAL_SRC_DIR, 'gt') # Fallback check for 'gt' folder
        print(f"Warning: Using 'gt' folder for labels from {VAL_SRC_DIR}. Ensure these are in YOLO format.")
    copy_files(val_src_img, val_src_lbl, VAL_IMG_DST, VAL_LBL_DST)

    # --- Test Data ---
    if os.path.exists(TEST_IMG_DST): shutil.rmtree(TEST_IMG_DST)
    if os.path.exists(TEST_LBL_DST): shutil.rmtree(TEST_LBL_DST)
    test_src_img = os.path.join(TEST_SRC_DIR, 'img1')
    test_src_lbl = os.path.join(TEST_SRC_DIR, 'labels') # ASSUMES labels are already in YOLO format in 'labels' folder
    if not os.path.exists(test_src_lbl):
        test_src_lbl = os.path.join(TEST_SRC_DIR, 'gt') # Fallback check for 'gt' folder
        print(f"Warning: Using 'gt' folder for labels from {TEST_SRC_DIR}. Ensure these are in YOLO format.")
    copy_files(test_src_img, test_src_lbl, TEST_IMG_DST, TEST_LBL_DST)

    # --- Final Validation Data ---
    if os.path.exists(FINAL_VAL_IMG_DST): shutil.rmtree(FINAL_VAL_IMG_DST)
    if os.path.exists(FINAL_VAL_LBL_DST): shutil.rmtree(FINAL_VAL_LBL_DST)
    final_val_src_img = os.path.join(FINAL_VAL_SRC_DIR, 'img1')
    final_val_src_lbl = os.path.join(FINAL_VAL_SRC_DIR, 'labels') # ASSUMES labels are already in YOLO format in 'labels' folder
    if not os.path.exists(final_val_src_lbl):
        final_val_src_lbl = os.path.join(FINAL_VAL_SRC_DIR, 'gt') # Fallback check for 'gt' folder
        print(f"Warning: Using 'gt' folder for labels from {FINAL_VAL_SRC_DIR}. Ensure these are in YOLO format.")
    copy_files(final_val_src_img, final_val_src_lbl, FINAL_VAL_IMG_DST, FINAL_VAL_LBL_DST)

    print("Dataset preparation finished.")


def create_data_yaml():
    """Creates the data.yaml file needed for YOLO training."""
    print(f"Creating dataset YAML file at: {DATA_YAML_PATH}")
    data_config = {
        'path': DATASET_DIR,  # Dataset root directory
        'train': os.path.join('train', 'images'),  # Relative path to train images
        'val': os.path.join('val', 'images'),      # Relative path to validation images
        'test': os.path.join('test', 'images'),    # Relative path to test images
        'final_val': os.path.join('final_val', 'images'), # Relative path to final validation images

        # --- Class names (IMPORTANT: Adjust if necessary!) ---
        'names': {
            0: 'ball',
            1: 'person' # Assuming class 1 is 'person' based on adasd.py and 000006.txt
            # Add other classes if your dataset has them, e.g., 2: 'referee'
        }
    }
    # Derive nc (number of classes) from names
    data_config['nc'] = len(data_config['names'])

    # Ensure the directory for the YAML file exists
    os.makedirs(os.path.dirname(DATA_YAML_PATH), exist_ok=True)

    with open(DATA_YAML_PATH, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    print("Dataset YAML created successfully.")
    return DATA_YAML_PATH

# --- Main Training Script ---
if __name__ == '__main__':
    print("Starting YOLO Training Script...")

    # 1. Prepare dataset structure (Copy files)
    #    IMPORTANT: This assumes your labels are already in YOLO format (.txt files)
    #    in a 'labels' or 'gt' subdirectory for each source dataset.
    #    If not, you need a separate script (like adasd.py) to convert them first.
    prepare_yolo_dataset()

    # 2. Create the data.yaml configuration file
    data_yaml = create_data_yaml()

    # 3. Initialize YOLO model
    model = YOLO("pre_trained.pt")  

    # 4. Train the model
    print(f"Starting training on device: {DEVICE}")
    print(f"Using data config: {data_yaml}")

    model.train(
        data=data_yaml,
        epochs=60,
        batch=BATCH_SIZE,
        imgsz=1024,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME + '_stage1_ball',
        exist_ok=True,
        freeze=[0,1,2,3],
        patience=15,
        augment=True,
        classes=[0],  # Only train on ball class
        # Learning rate settings
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,
        # Loss weights
        box=0.05,  # increased box loss weight for small objects
        cls=0.5,
        dfl=1.5,  # distribution focal loss weight
    )

    # Stage 2: full unfreeze, joint classes
    best = os.path.join(PROJECT_NAME, f"{RUN_NAME}_stage1_ball/weights/best.pt")
    model = YOLO(best)
    model.train(
        data=data_yaml,
        epochs=80,
        batch=BATCH_SIZE,
        imgsz=1536,              # higher resolution
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME + '_ball_only',
        exist_ok=True,
        patience=10,
        save_period=5,
        augment=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        classes=[0],  # Only train on ball class
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,
        # Loss weights
        box=0.05,  # increased box loss weight for small objects
        cls=0.5,
        dfl=1.5,  # distribution focal loss weight
    )

    # Last det beste fra steg 1
    best_model_path = os.path.join(PROJECT_NAME, RUN_NAME + "_ball_only", 'weights', 'best.pt')
    model = YOLO(best_model_path)

    # Tren videre på alt (ball + person)
    model.train(
        data=data_yaml,
        epochs=100,               # 30 ekstra runder (så totalt 50 epochs)
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME + "_full_train",
        exist_ok=True,
        patience=10,
        save_period=5,
        verbose=True,
    )


    print("Training finished.")
