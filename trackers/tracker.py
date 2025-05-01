from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append("../")
from utils import get_center_of_bbox, get_bbox_width
import cv2
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self, model_path='weights/best.pt'):
        # Initialize tracker with models
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            track_thresh=0.5,
            track_buffer=50,
            match_thresh=0.8,
            frame_rate=30
        )
        self.model_ball = YOLO("best_ball.pt")
        
    def detect_frames(self, frames):
        # Detect objects in batches of frames
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(
                source=frames[i:i+batch_size],
                conf=0.3, iou=0.5, imgsz=1024, classes=[1],
            ) 
            detections += batch
        return detections
    
    def interpolate_ball_positions(self, ball_positions):
        # Fill gaps in ball position data
        rows = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(rows, columns=['x1','y1','x2','y2'])
        df = df.interpolate().bfill()
        return [{1: {"bbox": bbox}} for bbox in df.to_numpy().tolist()]

    def get_object_tracks(self, frames, read_from_stuble=False, stuble_path=None, max_ball_distance=150):
        # Track players and ball through video frames
        if read_from_stuble and stuble_path and os.path.exists(stuble_path):
            with open(stuble_path, 'rb') as f:
                return pickle.load(f)

        raw_dets = self.detect_frames(frames)
        tracks = {"players": [], "ball": []}
        
        last_ball_pos = None 

        for idx, det in enumerate(raw_dets):
            names = det.names
            inv = {v:k for k,v in names.items()}
            person_cls = inv["person"]
            ball_cls   = inv["ball"]

            sup = sv.Detections.from_ultralytics(det)
            trk = self.tracker.update_with_detections(sup)

            tracks["players"].append({})
            tracks["ball"].append({})

            for bbox, cls_id, trk_id in zip(trk.xyxy, trk.class_id, trk.tracker_id):
                box = bbox.tolist()
                if cls_id == person_cls:
                    tracks["players"][idx][trk_id] = {"bbox": box}

            best_ball_score = -1
            best_ball_bbox = None
            
            all_detections = list(zip(sup.xyxy, sup.class_id, sup.confidence))
            for tracked in zip(trk.xyxy, trk.class_id, trk.confidence):
                all_detections.append(tracked)
                
            for bbox, cls_id, conf in all_detections:
                if cls_id == ball_cls:
                    current_score = conf
                    
                    if last_ball_pos is not None:
                        current_center = get_center_of_bbox(bbox.tolist())
                        distance = np.sqrt((current_center[0] - last_ball_pos[0])**2 + 
                                           (current_center[1] - last_ball_pos[1])**2)
                        
                        distance_factor = max(0, 1 - distance/max_ball_distance)
                        position_weight = 0.4
                        current_score = conf * (1-position_weight) + distance_factor * position_weight
                    
                    if current_score > best_ball_score:
                        best_ball_score = current_score
                        best_ball_bbox = bbox.tolist()
                        
            if best_ball_bbox is not None:
                tracks["ball"][idx][1] = {"bbox": best_ball_bbox}
                last_ball_pos = get_center_of_bbox(best_ball_bbox)
            
            if not tracks["ball"][idx]:
                frame = frames[idx]

                single = self.model_ball.predict(
                    source=frame,
                    conf=0.46,
                    iou=0.3,
                    imgsz=1920,
                    classes=[ball_cls],
                    augment=True,
                    agnostic_nms=True,
                )
                single_sup = sv.Detections.from_ultralytics(single[0])

                best_ball_score = -1
                best_ball_bbox = None

                for bbox, cls_id, conf in zip(single_sup.xyxy, single_sup.class_id, single_sup.confidence):
                    current_score = conf
                    
                    if last_ball_pos is not None:
                        current_center = get_center_of_bbox(bbox.tolist())
                        distance = np.sqrt((current_center[0] - last_ball_pos[0])**2 + 
                                          (current_center[1] - last_ball_pos[1])**2)
                        
                        distance_factor = max(0, 1 - distance/max_ball_distance)
                        position_weight = 0.4
                        current_score = conf * (1-position_weight) + distance_factor * position_weight
                    
                    if current_score > best_ball_score:
                        bx1, by1, bx2, by2 = bbox.tolist()
                        best_ball_score = current_score
                        best_ball_bbox = [bx1, by1, bx2, by2]

                if best_ball_bbox is not None:
                    tracks["ball"][idx][1] = {"bbox": best_ball_bbox}
                    last_ball_pos = get_center_of_bbox(best_ball_bbox)

        if stuble_path:
            with open(stuble_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # Draw ellipse for player
        y2 = int(bbox[3])
        x_c, _ = get_center_of_bbox(bbox)
        w = get_bbox_width(bbox)
        cv2.ellipse(frame, (int(x_c), y2), (int(w), int(w*0.35)),
                    0, -45, 235, color, 2, cv2.LINE_4)
        if track_id is not None:
            rw, rh = 40, 20
            x1 = int(x_c - rw/2); y1 = int(y2 - rh/2) + 15
            x2, y2r = x1 + rw, y1 + rh
            cv2.rectangle(frame, (x1, y1), (x2, y2r), color, cv2.FILLED)
            tx = x1 + 12 - (10 if track_id>99 else 0)
            cv2.putText(frame, f"{track_id}", (tx, y1+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        # Draw triangle for ball
        y, x_c = int(bbox[1]), get_center_of_bbox(bbox)[0]
        pts = np.array([[x_c, y], [x_c-10, y-20], [x_c+10, y-20]])
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0,0,0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        # Draw all annotations on video frames
        out = []
        for i, fr in enumerate(video_frames):
            f = fr.copy()
            for tid, p in tracks["players"][i].items():
                f = self.draw_ellipse(f, p["bbox"], (0,0,255), tid)
            for tid, b in tracks["ball"][i].items():
                f = self.draw_triangle(f, b["bbox"], (0,255,0))
            out.append(f)
        return out