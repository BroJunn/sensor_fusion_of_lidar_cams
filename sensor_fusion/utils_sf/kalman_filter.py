import cv2 
import numpy as np
from typing import List, Dict


from tools.visual_utils.open3d_vis_utils import translate_boxes_to_open3d_instance
from sensor_fusion.utils_sf.iou_3d import get_3d_box, box3d_iou
from sensor_fusion.utils_sf.iou_2d import compute_2d_iou

class KalmanFilter():

    def __init__(self, statePost, dt=0.1):
        self.dt = dt
        self.kf = cv2.KalmanFilter(11, 11)

        self._init_params()

        self.kf.statePost = statePost
        
    def _init_params(self):
        '''
        state: np.ndarray [x, y, z, vx, vy, vz, w, h, l, heading, rate_heading]
        obsevation: np.ndarray [x, y, z, vx, vy, vz, w, h, l, heading, rate_heading]
        '''
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, self.dt, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, self.dt, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, self.dt, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, self.dt],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kf.measurementMatrix = np.eye(11, dtype=np.float32)

        self.kf.processNoiseCov = np.eye(11, dtype=np.float32) * 1e-2

        self.kf.measurementNoiseCov = np.eye(11, dtype=np.float32) * 1e-2

        # self.kf.errorCovPost = np.eye(11, dtype=np.float32) * 1e4

        # self.kf.statePost = np.zeros(11, np.float32)
        
    def update(self, measurement):
        estimated = self.kf.correct(measurement)

        return estimated
    
    def predict(self):
        predicted = self.kf.predict()

        return predicted

class TrackedObject():
    def __init__(self):
        self.lost_count: int = 0
        self.state: np.ndarray 
        self.label_idx: int
        self.kalman_filter = None
        self.timestep: int = 0
        self.dt=0.1

        self.history: List[np.ndarray] = []
        self.label_history: List[int] = []

    def predict(self):
        return self.kalman_filter.predict()
    
    def update(self, detecition, label):
        if len(self.history) == 0:
            measurement = np.array([
                detecition[0], detecition[1], detecition[2], 0, 0, 0, detecition[3], detecition[4], detecition[5], detecition[6], 0
            ], dtype=np.float32)
            self.kalman_filter = KalmanFilter(measurement)
            self.state = measurement
        else:
            # measurement = np.array([
            #     detecition[0], detecition[1], detecition[2], 
            #     (detecition[0] - self.history[-1][0]) / self.dt, 
            #     (detecition[1] - self.history[-1][1]) / self.dt, 
            #     (detecition[2] - self.history[-1][2]) / self.dt, 
            #     detecition[3], detecition[4], detecition[5], detecition[6], (detecition[6] - self.history[-1][9]) / self.dt
            # ], dtype=np.float32)
            measurement = np.array([
                detecition[0], detecition[1], detecition[2], 
                (detecition[0] - self.history[-1][0]) / self.dt, 
                (detecition[1] - self.history[-1][1]) / self.dt, 
                (detecition[2] - self.history[-1][2]) / self.dt, 
                detecition[3], detecition[4], detecition[5],
                detecition[6] * 0.8 + np.arctan((detecition[1] - self.history[-1][1]) / (detecition[0] - self.history[-1][0])) * 0.2, 0.0
            ], dtype=np.float32)
            self.state = self.kalman_filter.update(measurement)
        self.history.append(self.state)
        self.label_history.append(label)
        self.timestep += 1
        self.lost_count = 0

class Trackers():
    def __init__(self,iou_threshold=0.18, max_lost=3):
        self.tracked_objs = []
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def each_frame(self, detections_info: Dict):
        detections: List[np.ndarray]
        labels: List[np.ndarray]
        detections = detections_info['detections']
        labels = detections_info['labels']
        predictions = [tracked_obj.predict() for tracked_obj in self.tracked_objs]

        matches, unmatched_detections, unmatched_predictions = self.data_association(predictions, detections, self.iou_threshold)
        
        for match in matches:
            tracker_idx, detection_idx = match
            self.tracked_objs[tracker_idx].update(detections[detection_idx], labels[detection_idx])

        for tracker_idx in sorted(unmatched_predictions, reverse=True):
            self.tracked_objs[tracker_idx].lost_count += 1
            if self.tracked_objs[tracker_idx].lost_count > self.max_lost:
                del self.tracked_objs[tracker_idx]

        for detection_idx in unmatched_detections:
            tracked_obj = TrackedObject()
            tracked_obj.update(detections[detection_idx], labels[detection_idx])
            self.tracked_objs.append(tracked_obj)

    def data_association(self, predictions: List[np.ndarray], detections: List[np.ndarray], iou_threshold: float):
        iou_matrix : np.ndarray
        iou_matrix = self._compute_iou_matrix(predictions, detections)
        matches = []
        unmatched_detections = []
        unmatched_predictions = []
        unmatched_predictions = np.arange(len(predictions), dtype=np.int32)
        unmatched_detections = np.arange(len(detections), dtype=np.int32)

        while iou_matrix.size > 0:
            max_iou_idx = np.argmax(iou_matrix)
            tracker_idx, detection_idx = np.unravel_index(max_iou_idx, iou_matrix.shape)

            if iou_matrix[tracker_idx, detection_idx] >= iou_threshold:
                matches.append((tracker_idx, detection_idx))
                iou_matrix[tracker_idx] = 0.0
                iou_matrix[:, detection_idx] = 0.0
                # iou_matrix = np.delete(iou_matrix, tracker_idx, axis=0)
                # iou_matrix = np.delete(iou_matrix, detection_idx, axis=1)
                unmatched_predictions = np.delete(unmatched_predictions, np.where(unmatched_predictions == tracker_idx)[0], axis=0)
                unmatched_detections = np.delete(unmatched_detections, np.where(unmatched_detections == detection_idx)[0], axis=0)
            else:
                break

        return matches, unmatched_detections, unmatched_predictions


    def _compute_iou_matrix(self, predictions: List[np.ndarray], detections: List[np.ndarray]):
        # iou_matrix = np.zeros((len(predictions), len(detections)), dtype=np.float32)
        iou_2d_matrix = np.zeros((len(predictions), len(detections)), dtype=np.float32)
        for i, prediction in enumerate(predictions):
            for j, detection in enumerate(detections):
                # iou_matrix[i, j] = self._compute_iou(np.squeeze(prediction), np.squeeze(detection))[0]
                iou_2d_matrix[i, j] = self._compute_iou(np.squeeze(prediction), np.squeeze(detection))
                
        return iou_2d_matrix

    def _compute_iou(self, prediction: np.ndarray, detection: np.ndarray) -> float:
        ''' compute iou between prediction and detection

        Input:
            prediction: np.ndarray [x, y, z, vx, vy, vz, w, h, l, heading, rate_heading]
            detection: np.ndarray [x, y, z, w, h, l, heading]
        Output:
            iou: 3D bounding box IoU
            iou_2d: bird's eye view 2D bounding box IoU

        '''
        # pred_box = get_3d_box(prediction[6:9], prediction[9].item(), prediction[:3])
        # det_box = get_3d_box(detection[3:6], detection[6].item(), detection[:3])
        # iou, iou_2d = box3d_iou(pred_box, det_box)

        iou_2d = compute_2d_iou(prediction[[0, 1, 6, 7, 9]], detection[[0, 1, 3, 4, 6]])
        return iou_2d
    
