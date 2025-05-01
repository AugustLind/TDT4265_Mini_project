from utils import save_image
import os
import cv2
from trackers import Tracker

def main():
    input_dir  = "RBK_TDT17/4_annotate_1min_bodo_start/img1"
    output_dir = "frames_output"

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    )

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    frames = []
    for p in image_files:
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: failed to read {p}")
            continue
        frames.append(img)

    tracker = Tracker("best.pt")

    tracks = tracker.get_object_tracks(
        frames,
        read_from_stuble=True,
        stuble_path="stubs/tracks_stuble.pkl"
    )

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    annotated_frames = tracker.draw_annotations(frames, tracks)

    saved_count = 0

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
