from utils import save_image
import os
import cv2
from trackers import Tracker

def main():
    input_dir  = "RBK_TDT17/4_annotate_1min_bodo_start/img1"
    output_dir = "frames_output"

    os.makedirs(output_dir, exist_ok=True)

    # get sorted full paths to all images
    image_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    )

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    # 1) load all images into memory first
    frames = []
    for p in image_files:
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: failed to read {p}")
            continue
        frames.append(img)

    tracker = Tracker("best.pt")

    # 2) Pass the actual frames (not paths) to get_object_tracks
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stuble=True,
        stuble_path="stubs/tracks_stuble.pkl"
    )

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 3) draw your annotations on the numpy arrays
    annotated_frames = tracker.draw_annotations(frames, tracks)

    saved_count = 0

    # 4) save each annotated frame, matching it back to its filename
    for idx, frame in enumerate(annotated_frames):
        base_filename = os.path.basename(image_files[idx])
        try:
            save_image(frame, output_dir, base_filename)
            saved_count += 1
        except Exception as e:
            print(f"Error saving image {base_filename}: {e}")

    print(f"{saved_count} images saved to {output_dir}")

if __name__ == "__main__":
    main()
