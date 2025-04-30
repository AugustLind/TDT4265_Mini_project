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
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(track_buffer=50)
        self.model_ball = YOLO("best_ball.pt")

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(
                source=frames[i:i+batch_size],
                conf=0.3, iou=0.5, imgsz=1024,
            )
            detections += batch
        return detections
    
    def interpolate_ball_positions(self, ball_positions):
        rows = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(rows, columns=['x1','y1','x2','y2'])
        df = df.interpolate().bfill()
        return [{1: {"bbox": bbox}} for bbox in df.to_numpy().tolist()]

    def get_object_tracks(self, frames, read_from_stuble=False, stuble_path=None):
        # 1) Optionally load from pickle
        if read_from_stuble and stuble_path and os.path.exists(stuble_path):
            with open(stuble_path, 'rb') as f:
                return pickle.load(f)

        # 2) First-pass multi-class detection + tracking
        raw_dets = self.detect_frames(frames)
        tracks = {"players": [], "ball": []}

        for idx, det in enumerate(raw_dets):
            # names & inverse
            names = det.names
            inv = {v:k for k,v in names.items()}
            person_cls = inv["person"]
            ball_cls   = inv["ball"]

            # supervision detections
            sup = sv.Detections.from_ultralytics(det)
            trk = self.tracker.update_with_detections(sup)

            tracks["players"].append({})
            tracks["ball"].append({})

            # assign tracked detections for players
            for bbox, cls_id, trk_id in zip(trk.xyxy, trk.class_id, trk.tracker_id):
                box = bbox.tolist()
                if cls_id == person_cls:
                    tracks["players"][idx][trk_id] = {"bbox": box}

            # For ball - find the best ball detection (highest confidence)
            best_ball_conf = -1
            best_ball_bbox = None
            
            # First check tracked balls
            for bbox, cls_id, conf, trk_id in zip(trk.xyxy, trk.class_id, trk.confidence, trk.tracker_id):
                if cls_id == ball_cls and conf > best_ball_conf:
                    best_ball_conf = conf
                    best_ball_bbox = bbox.tolist()
            
            # Also check untracked detections
            for bbox, cls_id, conf in zip(sup.xyxy, sup.class_id, sup.confidence):
                if cls_id == ball_cls and conf > best_ball_conf:
                    best_ball_conf = conf
                    best_ball_bbox = bbox.tolist()
                    
            # If we found a ball, add it with ID=1
            if best_ball_bbox is not None:
                tracks["ball"][idx][1] = {"bbox": best_ball_bbox}
            
            # 3) Second-pass if no ball found
            if not tracks["ball"][idx]:
                frame = frames[idx]
                h, w = frame.shape[:2]

                # --- compute crop window from previous frames' ball (if any) ---
                found_ball = False
                max_lookback = 30  # Maximum number of frames to look back
                
                if idx > 0:
                    # Look back through previous frames until we find a ball
                    for prev_idx in range(idx-1, max(idx-max_lookback-1, -1), -1):
                        if tracks["ball"][prev_idx] and 1 in tracks["ball"][prev_idx]:
                            prev = tracks["ball"][prev_idx][1]["bbox"]
                            x1, y1, x2, y2 = map(int, prev)
                            
                            # Calculate distance in frames and adjust expansion
                            frames_ago = idx - prev_idx
                            # Make search area larger the further back we go
                            expansion = min(250 + frames_ago * 10, 500)
                            
                            # expand by calculated amount, but clamp to image bounds
                            cx1 = max(0, x1 - expansion)
                            cy1 = max(0, y1 - expansion)
                            cx2 = min(w, x2 + expansion)
                            cy2 = min(h, y2 + expansion)
                            found_ball = True
                            break
                
                if not found_ball:
                    # fallback to entire frame if no previous ball found
                    cx1, cy1, cx2, cy2 = 0, 0, w, h

                crop = frame[cy1:cy2, cx1:cx2]

                # run ball-only detector on the cropped region
                single = self.model_ball.predict(
                    source=crop,
                    conf=0.15,
                    iou=0.3,
                    imgsz=1024,
                    classes=[ball_cls],
                    augment=True,
                    agnostic_nms=True,
                )
                single_sup = sv.Detections.from_ultralytics(single[0])

                # adjust any detections back to full-frame coords
                best_ball_conf = -1
                best_ball_bbox = None
                for bbox, cls_id, conf in zip(single_sup.xyxy, single_sup.class_id, single_sup.confidence):
                    if cls_id == ball_cls and conf > best_ball_conf:
                        bx1, by1, bx2, by2 = bbox.tolist()
                        # map from crop-coords → full-image coords
                        full_bbox = [
                            bx1 + cx1,
                            by1 + cy1,
                            bx2 + cx1,
                            by2 + cy1
                        ]
                        best_ball_conf = conf
                        best_ball_bbox = full_bbox

                if best_ball_bbox is not None:
                    tracks["ball"][idx][1] = {"bbox": best_ball_bbox}

        # 4) Optionally save
        if stuble_path:
            with open(stuble_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
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
        y, x_c = int(bbox[1]), get_center_of_bbox(bbox)[0]
        pts = np.array([[x_c, y], [x_c-10, y-20], [x_c+10, y-20]])
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0,0,0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        out = []
        for i, fr in enumerate(video_frames):
            f = fr.copy()
            for tid, p in tracks["players"][i].items():
                f = self.draw_ellipse(f, p["bbox"], (0,0,255), tid)
            for tid, b in tracks["ball"][i].items():
                f = self.draw_triangle(f, b["bbox"], (0,255,0))
            out.append(f)
        return out
