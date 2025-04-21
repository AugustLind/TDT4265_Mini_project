# Soccer Player Tracking System

This project implements a soccer player and ball tracking system for video footage using YOLOv8 object detection and ByteTrack tracking.

## Project Overview

The system can:
- Detect players, referees, and balls in soccer match footage
- Track objects across video frames to maintain consistent IDs
- Annotate video footage with tracking information
- Train custom YOLOv8 models on soccer footage

## Directory Structure

```
TDT4265_Mini_project/
├── models/                # Pre-trained model weights
├── RBK_TDT17/            # Dataset directory (not included in repo)
├── tasks/                # Core functionality 
│   ├── __init__.py
│   ├── detection.py      # Detection functionality
│   └── tracking.py       # Tracking functionality
├── main.py               # Main entry point for tracking
├── player_tracker.ipynb  # Jupyter notebook for testing/visualization
├── train_yolo.py         # Script for training YOLOv8 models
└── requirements.txt      # Project dependencies
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd TDT4265_Mini_project
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Data

#### Dataset
Download the RBK_TDT17 dataset and place it in the project root directory with the following structure:

```
RBK_TDT17/
├── 1_train-val_1min_aalesund_from_start/
│   ├── gt/
│   │   └── gt.txt        # Ground truth annotations
│   └── img1/
│       └── *.jpg         # Image frames
└── 2_train-val_1min_after_goal/
    ├── gt/
    │   └── gt.txt        # Ground truth annotations
    └── img1/
        └── *.jpg         # Image frames
```

#### Pre-trained Model
Either:
- Download a pre-trained model and save it as `models/best.pt`
- Train your own model using the `train_yolo.py` script

## Training a Custom Model

To train a custom YOLOv8 model on the soccer dataset:

```bash
python train_yolo.py
```

This will:
1. Prepare the dataset in YOLOv8 format
2. Create a dataset configuration file
3. Train a YOLOv8 model (default: 50 epochs)
4. Save the best model to `runs/soccer/player_ball_detection/weights/best.pt`

After training, copy the best model to the `models` directory:

```bash
mkdir -p models
cp runs/soccer/player_ball_detection/weights/best.pt models/
```

You can modify training parameters in `train_yolo.py` including:
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `img_size`: Image size for training
- `weights_path`: Initial weights to start from

## Running Object Tracking

To run the player tracking system:

```bash
python main.py
```

By default, this will:
1. Process images from `RBK_TDT17/1_train-val_1min_aalesund_from_start/img1`
2. Save annotated results to `tracked_output/`

To modify input/output paths, edit these variables in `main.py`:
```python
input_file_path = "path/to/input"  # Can be video file or folder with images
output_file_path = "path/to/output"  # Output folder or video file
```

## Working with Jupyter Notebook

For interactive exploration, use the included notebook:

```bash
jupyter notebook player_tracker.ipynb
```

The notebook demonstrates how to:
- Load the tracking model
- Process image sequences
- Visualize tracking results

## System Requirements

- Python 3.8+ 
- PyTorch 1.8+
- CUDA-compatible GPU recommended for training (but not required)
- Sufficient disk space for dataset and model weights (~1GB)

## Troubleshooting

- **CUDA errors**: Ensure your NVIDIA drivers are up to date
- **Model not found**: Check that you have the model file in `models/best.pt` 
- **Dataset errors**: Verify the RBK_TDT17 directory structure matches expectations
- **Memory issues**: Try reducing batch size in `train_yolo.py`
https://huggingface.co/uisikdag/yolo-v8-football-players-detection
https://huggingface.co/keremberke/yolov5n-football
