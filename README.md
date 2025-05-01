# Soccer Player and Ball Tracking Project

This project uses YOLOv8 object detection and ByteTrack tracking to identify and track players and the ball in soccer video frames. It processes image sequences, performs object detection and tracking, annotates the frames with bounding boxes and track IDs, and saves the annotated frames. Optionally, it can create a video from these frames.

## Project Structure

├── create_video.py       # Script to assemble frames into video
├── main.py               # Main script for tracking and annotation
├── trackers/             # Tracking functionality
│   ├── __init__.py
│   └── tracker.py        # Tracker class implementation
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── bbox_utils.py     # Bounding box utilities
│   └── img_utils.py      # Image handling utilities
├── yolo_training.py      # Script for YOLO model training
├── best.pt               # Trained player detection model
├── best_ball.pt          # Specialized ball detection model
├── pre_trained.pt        # Pre-trained model for fine-tuning
├── data.yaml             # Dataset configuration
├── requirements.txt      # Project dependencies
└── frames_output/        # Directory for annotated output frames

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:** t)
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data:** Ensure the raw data is present in the `RBK_TDT17` directory as expected by the scripts, or modify the paths in the scripts. You may use your own data instead.

## Usage

### 1. Training the Model (Optional)

If you want to train the YOLO model from scratch or fine-tune it:

1.  **Prepare Data:** The script assumes raw data is in `RBK_TDT17` with `img1` and `labels` (or `gt`) subdirectories.
2.  **Run Training:**
    ```bash
    python yolo_training.py
    ```
    This script will:
    *   Copy and organize images/labels into the [dataset](http://_vscodecontentref_/18) folder.
    *   Create the [soccer_data.yaml](http://_vscodecontentref_/19) file.
    *   Train the YOLO model using [yolov8m.pt](http://_vscodecontentref_/20) as a base, saving results to [soccer_training](http://_vscodecontentref_/21).
    *   The best model weights will be saved, typically as `runs/soccer_training/exp_full_train/weights/best.pt`. You might need to copy this to [best.pt](http://_vscodecontentref_/22) in the root directory for the main script.

### 2. Running Tracking and Annotation

1.  **Configure Input/Output:** Modify [input_dir](http://_vscodecontentref_/23) and [output_dir](http://_vscodecontentref_/24) in [main.py](http://_vscodecontentref_/25) if necessary. By default, it uses [img1](http://_vscodecontentref_/26) as input and saves annotated frames to `frames_output`.
2.  **Run the main script:**
    ```bash
    python main.py
    ```
    This will:
    *   Load images from the [input_dir](http://_vscodecontentref_/27).
    *   Use the [Tracker](http://_vscodecontentref_/28) class with the model weights ([best.pt](http://_vscodecontentref_/29)) to detect and track objects. It can optionally load pre-computed tracks from [tracks_stuble.pkl](http://_vscodecontentref_/30).
    *   Interpolate ball positions.
    *   Draw annotations (ellipses for players, triangles for the ball) on the frames.
    *   Save the annotated frames to the [output_dir](http://_vscodecontentref_/31).

### 3. Creating a Video (Optional)

1.  **Ensure annotated frames exist:** Run [main.py](http://_vscodecontentref_/32) first to generate frames in the `frames_output` directory (or the directory specified in [create_video.py](http://_vscodecontentref_/33)).
2.  **Run the video creation script:**
    ```bash
    python create_video.py
    ```
    This will create [output_video.mp4](http://_vscodecontentref_/34) from the `.jpg` or `.png` frames found in the `frames_output` directory.

## Key Files

*   [main.py](http://_vscodecontentref_/35): Entry point for running the tracking and annotation process on image frames.
*   [tracker.py](http://_vscodecontentref_/36): Contains the [Tracker](http://_vscodecontentref_/37) class responsible for object detection (YOLO) and tracking (ByteTrack).
*   [yolo_training.py](http://_vscodecontentref_/38): Script for preparing the dataset and training the YOLOv8 model.
*   [create_video.py](http://_vscodecontentref_/39): Script to assemble annotated frames into a video file.
*   [soccer_data.yaml](http://_vscodecontentref_/40): Configuration file defining dataset paths and class names for YOLO training.
*   [best.pt](http://_vscodecontentref_/41): Trained model weights used by [main.py](http://_vscodecontentref_/42).

## License

This project is licensed under the MIT License - see the [LICENSE](http://_vscodecontentref_/43) file for details.
