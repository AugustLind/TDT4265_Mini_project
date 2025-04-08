from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import cv2
import glob

system_directory = os.getcwd()
model_path = f"{system_directory}/runs/soccer/player_ball_detection/weights/best.pt"

ellipse_annotator = sv.EllipseAnnotator()
label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    text_padding=3,
    color=sv.Color.black(),
)
model = YOLO(model_path)
model.set_classes = (["player", "referee", "ball"])

class LimitedTracker:
    """Custom tracker wrapper to maintain consistent IDs with limits for each class"""
    def __init__(self, max_players=22, max_referees=1, max_balls=1):
        self.tracker = sv.ByteTrack()
        self.max_persons = max_players + max_referees
        self.max_balls = max_balls
        
        self.active_person_ids = set()
        self.active_ball_ids = set()
        
    def update_with_detections(self, detections):
        tracked_detections = self.tracker.update_with_detections(detections)
        
        if tracked_detections.tracker_id is None:
            return tracked_detections
            
        mask = np.ones(len(tracked_detections), dtype=bool)
        
        for i, (class_id, track_id) in enumerate(zip(tracked_detections.class_id, tracked_detections.tracker_id)):
            if class_id == 0 or class_id == 1:
                if track_id in self.active_person_ids:
                    mask[i] = True
                elif len(self.active_person_ids) < self.max_persons:
                    self.active_person_ids.add(track_id)
                    mask[i] = True
                else:
                    mask[i] = False
            elif class_id == 2:
                if track_id in self.active_ball_ids:
                    mask[i] = True
                elif len(self.active_ball_ids) < self.max_balls:
                    self.active_ball_ids.add(track_id)
                    mask[i] = True
                else:
                    mask[i] = False
        
        return tracked_detections[mask]
    
    def reset(self):
        self.tracker = sv.ByteTrack()
        self.active_person_ids.clear()
        self.active_ball_ids.clear()


tracker = LimitedTracker(max_players=22, max_referees=1, max_balls=1)

class TrackAndAnnotate:
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        
        self.person_annotator = sv.EllipseAnnotator(
            color=sv.Color.red(),
            thickness=2
        )
        self.ball_annotator = sv.EllipseAnnotator(
            color=sv.Color.blue(),
            thickness=3,
        )
        
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=3,
            color=sv.Color.black(),
        )

    def callback(self, frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        confidence_threshold = 0.3
        if detections.confidence is not None:
            mask = detections.confidence > confidence_threshold
            detections = detections[mask]
        
        detections = tracker.update_with_detections(detections)
        
        annotated_frame = frame.copy()
        
        if len(detections) > 0:
            if detections.class_id is not None:
                person_mask = (detections.class_id == 0) | (detections.class_id == 1)
                ball_mask = detections.class_id == 2
                
                if any(person_mask):
                    person_detections = detections[person_mask]
                    annotated_frame = self.person_annotator.annotate(annotated_frame, person_detections)
                
                if any(ball_mask):
                    ball_detections = detections[ball_mask]
                    annotated_frame = self.ball_annotator.annotate(annotated_frame, ball_detections)
        
        if detections.tracker_id is not None:
            labels = []
            for i, class_id in enumerate(detections.class_id):
                track_id = detections.tracker_id[i]
                if class_id == 0 or class_id == 1:
                    labels.append(f"P#{track_id}")
                elif class_id == 2:
                    labels.append(f"Ball#{track_id}")
                else:
                    labels.append(f"#{track_id}")
            
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections,
                labels=labels
            )
            
        return annotated_frame

    def track_and_anotate(self):
        tracker.reset()
        
        if os.path.isdir(self.input_file_path):
            self.track_and_annotate_images()
        else:
            sv.process_video(
                source_path=self.input_file_path,
                target_path=self.output_file_path,
                callback=self.callback
            )
    
    def track_and_annotate_images(self):
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_file_path, f'*.{ext}')))
        
        image_files.sort()
        
        if not image_files:
            print(f"No image files found in {self.input_file_path}")
            return
        
        os.makedirs(self.output_file_path, exist_ok=True)
        
        for i, image_path in enumerate(image_files):
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to read image: {image_path}")
                continue
            
            processed_frame = self.callback(frame, i)
            
            filename = os.path.basename(image_path)
            output_path = os.path.join(self.output_file_path, filename)
            cv2.imwrite(output_path, processed_frame)
            
            if i % 50 == 0 or i == len(image_files) - 1:
                print(f"Processed {i+1}/{len(image_files)}: {filename}")