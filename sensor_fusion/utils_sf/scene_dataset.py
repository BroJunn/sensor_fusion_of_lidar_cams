from pcdet.datasets import DatasetTemplate
from utils_sf.utils import get_K_disCoef, get_R_t, R_t_to_T

import pandas as pd
import cv2
import os, sys
import numpy as np

class SceneDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.scene_dir_list = os.listdir(root_path)
        
        self.cam_names = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right', 'stereo_front_left', 'stereo_front_right']

    def __len__(self):
        return len(self.self.scene_dir_list)

    def __getitem__(self, index):
        
        scene_path = os.path.join(self.root_path, self.scene_dir_list[index])
        intrinsics = {}
        for cam_name in self.cam_names:
            intrinsics[cam_name] = self._load_cam_intrinsics(scene_path, cam_name)

        extrinsics = {}
        for cam_name in self.cam_names:
            extrinsics[cam_name] = self._load_extrinsic(scene_path, cam_name, T_matrix=True)

        lidar_timestamps = self._get_lidar_timestamp(scene_path)
        cam_timestamps = {}
        for cam_name in self.cam_names:
            cam_timestamps[cam_name] = self._get_cam_timestamp(scene_path, cam_name)

        ego_se3_dataframe, timestamps_ego = self._load_ego_SE3(scene_path)

        list_lidar_data_dict = []
        for lidar_timestamp in lidar_timestamps:
            input_dict = {
                'points': self._load_lidar(scene_path, lidar_timestamp),
                'frame_id': index,
            }
            data_dict = self.prepare_data(data_dict=input_dict)
            list_lidar_data_dict.append(data_dict)

        cam_data_dict = {}
        for cam_name in self.cam_names:
            cam_data_list = []
            for cam_timestamp in cam_timestamps[cam_name]:
                cam_data_list.append(self._load_img(scene_path, cam_name, cam_timestamp))
            cam_data_dict[cam_name] = cam_data_list

        return {'lidar_data': list_lidar_data_dict,
                'lidar_timestamps': lidar_timestamps,
                'cam_data': cam_data_dict,
                'cam_timestamps': cam_timestamps,
                'ego_se3_dataframe': ego_se3_dataframe,
                'timestamps_ego': timestamps_ego,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,}

    def _get_lidar_timestamp(self, scene_path):
        list_lidar_file = os.listdir(os.path.join(scene_path, 'sensors/lidar/'))
        list_lidar_timestamp = [int(lidar_file.split('.')[0]) for lidar_file in list_lidar_file]

        return sorted(list_lidar_timestamp)

    def _get_cam_timestamp(self, scene_path, sensor_name):
        list_cam_file = os.listdir(os.path.join(scene_path, 'sensors/cameras/{}/'.format(sensor_name)))
        list_cam_timestamp = [int(cam_file.split('.')[0]) for cam_file in list_cam_file]

        return sorted(list_cam_timestamp)

    def _load_ego_SE3(self, scene_path):
        ego_se3 = pd.read_feather(os.path.join(scene_path, 'city_SE3_egovehicle.feather'))
        timestamps = ego_se3['timestamp_ns']
        timestamps = sorted(timestamps)

        return ego_se3, timestamps

    @staticmethod
    def get_ex_from_ego_SE3(ego_se3_dataframe: pd.DataFrame, timestamp: int, T_matrix: bool):
        R, t = get_R_t(ego_se3_dataframe, timestamp, idx_name='timestamp_ns')
        if T_matrix:
            return R_t_to_T(R, t)
        else:
            return R, t

    def _load_img(self, scene_path, sensor_name, timestamp):
        img = cv2.imread(os.path.join(scene_path, 'sensors/cameras/{}/{}.jpg'.format(sensor_name, timestamp)), cv2.IMREAD_GRAYSCALE)
        return img

    def _load_cam_intrinsics(self, scene_path, sensor_name):
        """ Load camera intrinsics from the dataset """
        intr_info = pd.read_feather(os.path.join(scene_path, 'calibration/intrinsics.feather'))
        K, disCoef = get_K_disCoef(intr_info, sensor_name)

        return K, disCoef

    def _load_extrinsic(self, scene_path, sensor_name, T_matrix: bool) -> np.ndarray:
        """ Load extrinsic from the dataset """
        ex_sensors = pd.read_feather(os.path.join(scene_path, 'calibration/egovehicle_SE3_sensor.feather'))
        R, t = get_R_t(ex_sensors, sensor_name)
        if T_matrix:
            return R_t_to_T(R, t)
        else:
            return R, t

    def _load_lidar(self, scene_path, timestamp) -> np.ndarray:
        """ Load lidar point cloud from the dataset """
        pointcloud = pd.read_feather(os.path.join(scene_path, 'sensors/lidar/{}.feather'.format(timestamp)))
        pointcloud_pos = pointcloud[['x', 'y', 'z']].to_numpy()
        intensity = pointcloud['intensity'].to_numpy()
        intensity_norm = intensity / np.max(intensity)

        return np.hstack([pointcloud_pos, intensity_norm[:, None]])
