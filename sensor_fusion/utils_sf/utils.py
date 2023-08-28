import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation
from tools.visual_utils.open3d_vis_utils import translate_boxes_to_open3d_instance
import torch
from typing import Union, Dict

def rotate_heading(heading_angle, rotation_matrix):
    unit_vector = np.vstack([np.cos(heading_angle), np.sin(heading_angle), np.zeros_like(heading_angle)])
    rotated_vector = np.dot(rotation_matrix, unit_vector)
    rotated_heading = np.arctan2(rotated_vector[1], rotated_vector[0])
    
    return rotated_heading

def transform_3d_detection_to_world_frame(pred_boxes, T):
    # pred_boxes [N, 7]
    # T: np.ndarray [4, 4]
    if isinstance(T, torch.Tensor):
        T = T.cpu().numpy()
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.cpu().numpy()
    points = pred_boxes[:, :3]
    points_world = np.dot(T, coor_to_homogeneous(points).T).T
    points_world = points_world[:, :3]

    heading_world = rotate_heading(pred_boxes[:, 6], T[:3, :3])
    
    return np.hstack([points_world[:, :3], pred_boxes[:, 3:6], heading_world[:, None]])

def find_closest_timestamps_sorted(list1, list2):
    ### find the closest timestamps in list2 for each timestamp in list1 ###

    i, j = 0, 0
    closest_indices = []
    closest_timestamps = []

    while i < len(list1):
        closest_distance = abs(list1[i] - list2[j])
        closest_idx = j

        while j + 1 < len(list2) and abs(list1[i] - list2[j+1]) <= closest_distance:
            j += 1
            closest_distance = abs(list1[i] - list2[j])
            closest_idx = j

        closest_indices.append(closest_idx)
        closest_timestamps.append(list2[closest_idx])
        i += 1

        if j > 0:
            j -= 1

    diffs = np.array(list1) - np.array(closest_timestamps)
    return closest_indices, closest_timestamps, diffs

def get_K_disCoef(intr_info: pd.DataFrame,
                    name_cam: str):
    intr_cam = intr_info.loc[intr_info['sensor_name'] == name_cam, ['fx_px', 'fy_px', 'cx_px', 'cy_px', 'k1', 'k2', 'k3', 'height_px', 'width_px']]
    K = np.array([[intr_cam['fx_px'].item(), 0, intr_cam['cx_px'].item()], [0, intr_cam['fy_px'].item(), intr_cam['cy_px'].item()], [0, 0, 1]])
    disCoef = np.array([intr_cam['k1'].item(), intr_cam['k2'].item(), 0.0, 0.0, intr_cam['k3'].item()])
    return K, disCoef

def get_R_t(extr_info: pd.DataFrame,
             name_cam: str,
             idx_name: str = 'sensor_name'):
    q_cam = extr_info.loc[extr_info[idx_name] == name_cam, ['qw', 'qx', 'qy', 'qz']]
    t_cam = extr_info.loc[extr_info[idx_name] == name_cam, ['tx_m', 'ty_m', 'tz_m']]
    q = q_cam.to_numpy().squeeze(0)
    rotation = Rotation.from_quat(q)
    R = rotation.as_matrix()
    T = np.array([t_cam['tx_m'].item(), t_cam['ty_m'].item(), t_cam['tz_m'].item()])
    return R, T

def calculate_baseline(T1, T2, R1, R2):
    # 计算一个相机相对于另一个的旋转和平移
    R_rel = np.dot(R2, np.linalg.inv(R1))
    T_rel = T2 - np.dot(R_rel, T1)
    
    return R_rel, T_rel

def get_line_set(boxes: Union[torch.Tensor, np.ndarray]):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    list_line_set = []
    for i in range(boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(boxes[i])
        list_line_set.append(line_set)
    return list_line_set
    
def R_t_to_T(R, t):
    m = np.hstack([R, t.reshape(3, 1)])
    return np.vstack([m, np.array([0, 0, 0, 1]).reshape(1, 4)])

def coor_to_homogeneous(coors: np.ndarray):
    assert coors.ndim == 2
    assert coors.shape[0] >= 1 and coors.shape[1] == 3
    
    ones = np.ones((coors.shape[0], 1))
    return np.hstack([coors, ones])