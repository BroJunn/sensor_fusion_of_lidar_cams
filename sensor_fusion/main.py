import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from utils_sf.utils import (
    transform_3d_detection_to_world_frame,
    find_closest_timestamps_sorted,
)
from utils_sf.scene_dataset import SceneDataset
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file
from ultralytics import YOLO
import cv2
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import glob
import argparse
from tqdm import tqdm
from utils_sf.kalman_filter import KalmanFilter, TrackedObject, Trackers
from utils_sf.visualize_tracker import TrackerVisualizer
from utils_sf.detection_2d_utils import Detect2dFilter
import copy

try:
    import open3d
    from tools.visual_utils import open3d_vis_utils as V

    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V

    OPEN3D_FLAG = False


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="tools/cfgs/argo2_models/cbgs_voxel01_voxelnext.yaml",
        help="specify the config for demo",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/yujun/Code/pc_det/pc_data_npy/data4.npy",
        help="specify the point cloud data file or directory",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/yujun/Code/pc_det/pth_folder/VoxelNeXt_Argo2_arranged.pth",
        help="specify the pretrained model",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".npy",
        help="specify the extension of your point cloud data file",
    )

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info("-----------------Quick Demo of Sensor Fusion-------------------------")

    scene_dataset = SceneDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path="/home/yujun/Dataset/Argoverse2_sensor/train-000/sensor/train/",
    )

    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=scene_dataset
    )
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # 2d detection model
    # model = YOLO("yolov8n.pt").cuda()
    # model.eval()
    # results = model("")

    with torch.no_grad():
        for scene_idx, scene_info_dict in enumerate(scene_dataset):
            # load data
            lidar_datas = scene_info_dict["lidar_data"]
            timestamps_lidar = scene_info_dict["lidar_timestamps"]
            ego_se3_dataframe = scene_info_dict["ego_se3_dataframe"]
            timestamps_ego = scene_info_dict["timestamps_ego"]

            # init tracker and visualizer
            tracker = Trackers()
            (
                closest_indices_lidar_ego,
                closest_timestamps_lidar_ego,
                diffs_lidar_ego,
            ) = find_closest_timestamps_sorted(timestamps_lidar, timestamps_ego)
            vis_tracker = TrackerVisualizer()
            detect_2d_filter = Detect2dFilter(scene_info_dict["cam_data"], scene_info_dict["cam_timestamps"], timestamps_lidar, scene_dataset.cam_names, scene_info_dict["intrinsics"], scene_info_dict["extrinsics"])
            # flag_first_frame = True

            # iterate through each lidar frame
            for idx_lidar, lidar_data in tqdm(enumerate(lidar_datas)):
                # inference 3d detection
                data_dict = scene_dataset.collate_batch([lidar_data])
                load_data_to_gpu(data_dict)
                pred_dict, _ = model.forward(data_dict)
                filtered_dict = {
                    "pred_scores": pred_dict[0]["pred_scores"][
                        pred_dict[0]["pred_scores"] >= 0.5
                    ],
                    "pred_boxes": pred_dict[0]["pred_boxes"][
                        pred_dict[0]["pred_scores"] >= 0.5
                    ],
                    "pred_labels": pred_dict[0]["pred_labels"][
                        pred_dict[0]["pred_scores"] >= 0.5
                    ],
                }
                filtered_dict_2d_detect = detect_2d_filter.filter(idx_lidar, filtered_dict)
                # view 3d point clouds
                # V.draw_scenes(
                #     points=data_dict['points'][:, 1:], ref_boxes=pred_dict[0]['pred_boxes'],
                #     ref_scores=pred_dict[0]['pred_scores'], ref_labels=pred_dict[0]['pred_labels']
                # )

                # view filtered 3d detection
                # V.draw_scenes(
                #     points=data_dict['points'][:, 1:], ref_boxes=filtered_dict['pred_boxes'],
                #     ref_scores=filtered_dict['pred_scores'], ref_labels=filtered_dict['pred_labels']
                # )
                T_ref = SceneDataset.get_ex_from_ego_SE3(
                    ego_se3_dataframe, timestamps_ego[0], True
                )
                T_ego = SceneDataset.get_ex_from_ego_SE3(
                    ego_se3_dataframe,
                    timestamps_ego[closest_indices_lidar_ego[idx_lidar]],
                    True,
                )
                filtered_dict_ref = {
                    "pred_scores": filtered_dict_2d_detect["pred_scores"],
                    "pred_boxes": transform_3d_detection_to_world_frame(
                        filtered_dict_2d_detect["pred_boxes"], np.linalg.inv(T_ref) @ T_ego
                    ),
                    "pred_labels": filtered_dict_2d_detect["pred_labels"],
                }
                # if flag_first_frame == True:
                #     filtered_dict_ref_all = copy.deepcopy(filtered_dict_ref)
                #     flag_first_frame = False
                # else:
                #     filtered_dict_ref_all["pred_scores"] = torch.cat(
                #         (
                #             filtered_dict_ref_all["pred_scores"],
                #             filtered_dict_ref["pred_scores"],
                #         ),
                #         0,
                #     )
                #     filtered_dict_ref_all["pred_boxes"] = np.concatenate(
                #         (
                #             filtered_dict_ref_all["pred_boxes"],
                #             filtered_dict_ref["pred_boxes"],
                #         ),
                #         0,
                #     )
                #     filtered_dict_ref_all["pred_labels"] = torch.cat(
                #         (
                #             filtered_dict_ref_all["pred_labels"],
                #             filtered_dict_ref["pred_labels"],
                #         ),
                #         0,
                #     )
                # V.draw_scenes(
                #     points=data_dict['points'][:, 1:], ref_boxes=filtered_dict_ref_all['pred_boxes'],
                #     ref_scores=filtered_dict_ref_all['pred_scores'], ref_labels=filtered_dict_ref_all['pred_labels']
                # )
                tracker.each_frame(
                    {
                        "detections": list(filtered_dict_ref["pred_boxes"]),
                        "labels": list(filtered_dict_ref["pred_labels"]),
                    }
                )
                vis_tracker.visualize(tracker)
            vis_tracker.generate_anim('ani_with_2d_filter_' + str(scene_idx) + '.gif')

# visualize the result
if __name__ == "__main__":
    main()
