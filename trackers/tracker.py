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
        self.tracker = sv.ByteTrack(track_buffer=30)

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(
                source=frames[i:i+batch_size],
                conf=0.2, iou=0.5, imgsz=1024
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

            # assign tracked detections
            for bbox, cls_id, trk_id in zip(trk.xyxy, trk.class_id, trk.tracker_id):
                box = bbox.tolist()
                if cls_id == person_cls:
                    tracks["players"][idx][trk_id] = {"bbox": box}
                elif cls_id == ball_cls:
                    tracks["ball"][idx][trk_id]   = {"bbox": box}

            # also ensure we at least get any ball as ID=1
            for bbox, cls_id in zip(sup.xyxy, sup.class_id):
                if cls_id == ball_cls:
                    tracks["ball"][idx][1] = {"bbox": bbox.tolist()}

            # 3) Second-pass if no ball found
            if not tracks["ball"][idx]:
                single = self.model.predict(
                    source=frames[idx],
                    conf=0.2, iou=0.5, imgsz=1024,
                    classes=[ball_cls]
                )
                single_sup = sv.Detections.from_ultralytics(single[0])
                for bbox, cls_id in zip(single_sup.xyxy, single_sup.class_id):
                    if cls_id == ball_cls:
                        tracks["ball"][idx][1] = {"bbox": bbox.tolist()}
                        break

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
