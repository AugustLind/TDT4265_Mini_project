import cv2
import os
import glob

def create_video_from_frames(input_dir, output_video_path, fps=30):
    """
    Creates a video from image frames in a directory.

    Args:
        input_dir (str): Directory containing the image frames.
        output_video_path (str): Path to save the output video file.
        fps (int): Frames per second for the output video.
    """
    image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg'))) # Assuming jpg files, adjust if needed
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(input_dir, '*.png'))) # Try png if no jpg found
    
    if not image_files:
        print(f"Error: No image files found in {input_dir}")
        return

    # Read the first image to get frame dimensions
    try:
        first_frame = cv2.imread(image_files[0])
        if first_frame is None:
            print(f"Error: Could not read the first image: {image_files[0]}")
            return
        height, width, layers = first_frame.shape
        size = (width, height)
    except Exception as e:
        print(f"Error reading first frame dimensions: {e}")
        return

    # Initialize VideoWriter
    # Use 'mp4v' codec for .mp4 file format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    if not out.isOpened():
        print(f"Error: Could not open video writer for path {output_video_path}")
        return

    print(f"Starting video creation: {output_video_path} with {len(image_files)} frames.")
    
    # Write frames to video
    for i, filename in enumerate(image_files):
        try:
            frame = cv2.imread(filename)
            if frame is None:
                print(f"Warning: Skipping file, could not read image: {filename}")
                continue
            # Ensure frame dimensions match the video size
            if frame.shape[1] != width or frame.shape[0] != height:
                print(f"Warning: Resizing frame {filename} from {frame.shape[1]}x{frame.shape[0]} to {width}x{height}")
                frame = cv2.resize(frame, size)
            out.write(frame)
            # Optional: print progress
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} frames...")
        except Exception as e:
            print(f"Error processing frame {filename}: {e}")
            continue

    # Release everything when job is finished
    out.release()
    print(f"Video saved successfully to {output_video_path}")

if __name__ == "__main__":
    input_directory = "frames_output"
    output_video_file = "output_video.mp4"
    frames_per_second = 30  # Adjust FPS as needed

    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.")
    else:
        create_video_from_frames(input_directory, output_video_file, frames_per_second)