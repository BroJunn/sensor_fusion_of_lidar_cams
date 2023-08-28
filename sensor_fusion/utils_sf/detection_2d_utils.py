from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import torch

from utils_sf.iou_2d import compute_2d_iou
from utils_sf.utils import (
    find_closest_timestamps_sorted,
    get_line_set,
    coor_to_homogeneous,
)


class Detect2dFilter:
    def __init__(
        self,
        cam_data_dict,
        cam_timestamps,
        lidar_timestamps,
        cam_names,
        intrinsics,
        extrinsics,
    ):
        """
        cam_data_dict: Dict[str, List[np.ndarray]]
        cam_timestamps: Dict[str, List[int]]
        lidar_timestamps: List[int]
        cam_names: List[str]
        intrinsics: Dict[str, np.ndarray]
        extrinsics: Dict[str, np.ndarray]
        """
        self.cam_data_dict = cam_data_dict
        self.cam_timestamps = cam_timestamps
        self.lidar_timestamps = lidar_timestamps
        self.cam_names = cam_names
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.closest_timestamps_cams = {}
        self.closest_indices_cams = {}
        self.diffs_cams = {}
        for cam_name in self.cam_names:
            (
                self.closest_indices_cams[cam_name],
                self.closest_timestamps_cams[cam_name],
                self.diffs_cams[cam_name],
            ) = find_closest_timestamps_sorted(
                lidar_timestamps, cam_timestamps[cam_name]
            )

        self.model = YOLO("sensor_fusion/yolo_pt/yolov8n.pt")
        self.model.model.eval()
        self.model.model.cuda()
        print()

    def filter(self, idx_lidar: int, pred_dict):
        list_line_set = get_line_set(pred_dict["pred_boxes"])
        valid_pred_idx = []
        for cam_name in self.cam_names:
            T = self.extrinsics[cam_name]
            K = self.intrinsics[cam_name][0]
            img = self.cam_data_dict[cam_name][
                self.closest_indices_cams[cam_name][idx_lidar]
            ]
            results = self.model(img[:, :, None].repeat(3, axis=2))
            r = results[0]

            # self._vis_2d_detection_result(r)
            xywhs = r.boxes.xywh
            # img_2d = self._draw_2d_bbox_on_image(img.copy(), xywhs)
            # cv2.imwrite('2d_bbox.png', img_2d)

            # vis 3d bbox
            # img_3d = self._draw_3d_bbox_on_image(img.copy(), list_line_set, T, K)
            # cv2.imwrite('3d_bbox.png', img_3d)
            list_points_2d, valid_list_line_set = self._3dbbox_to_2dbbox(
                list_line_set, T, K, img.shape
            )
            valid_idx = self._filter_valid_3d_detect(
                xywhs, list_points_2d, valid_list_line_set
            )
            valid_pred_idx = self._merge_list1_to_list2(valid_idx, valid_pred_idx)
        valid_pred_idx = sorted(valid_pred_idx)

        # rearrange pred_dict
        return {
            "pred_scores": pred_dict["pred_scores"][valid_pred_idx],
            "pred_boxes": pred_dict["pred_boxes"][valid_pred_idx],
            "pred_labels": pred_dict["pred_labels"][valid_pred_idx],
        }

    def _merge_list1_to_list2(self, list1, list2):
        """
        list1: List
        list2: List

        return: List
        """
        list2.extend(list1)

        unique_elements = []
        for elem in list2:
            if elem not in unique_elements:
                unique_elements.append(elem)

        return unique_elements

    def _vis_2d_detection_result(self, yolo_res):
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])
        im.save("2d_detect_res.jpg")

    def _draw_2d_bbox_on_image(self, img, xywhs):
        """
        img: np.ndarray with shape [H, W, 3]
        xywhs: List[np.ndarray] with shape [N, 4]
        """
        for xywh in xywhs:
            x, y, w, h = xywh
            top_left = (int(x - w // 2), int(y - h // 2))
            bottom_right = (int(x + w // 2), int(y + h // 2))

            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        return img

    def _draw_3d_bbox_on_image(self, img, list_line_set, T, K):
        list_points_2d, _ = self._3dbbox_to_2dbbox(list_line_set, T, K, img.shape)
        line_paris = np.asarray(list_line_set[0].lines)
        for points_2d in list_points_2d:
            for line_pair in line_paris:
                img = cv2.line(
                    img,
                    (int(points_2d[line_pair[0], 0]), int(points_2d[line_pair[0], 1])),
                    (int(points_2d[line_pair[1], 0]), int(points_2d[line_pair[1], 1])),
                    (0, 0, 255),
                    1,
                )

        return img

    def _3dbbox_to_2dbbox(self, list_line_set, T, K, img_size):
        list_points_2d = []
        valid_list_line_set = []
        for idx_3d_det, line_set in enumerate(list_line_set):
            box_points = np.asarray(line_set.points)
            box_points_homo = coor_to_homogeneous(box_points)
            box_points_homo_cam = np.dot(np.linalg.inv(T), box_points_homo.T).T
            box_points_cam = (
                box_points_homo_cam[:, :3] / box_points_homo_cam[:, 3][:, None]
            )
            points_2d = np.dot(K, box_points_cam.T).T
            points_2d = points_2d[:, :2] / points_2d[:, 2][:, None]
            if not (
                points_2d.min() < 0
                or points_2d[:, 0].max() >= img_size[0]
                or points_2d[:, 1].max() >= img_size[1]
            ):
                list_points_2d.append(points_2d)
                valid_list_line_set.append(idx_3d_det)

        return list_points_2d, valid_list_line_set

    def _filter_valid_3d_detect(
        self, xywhs, list_points_2d, valid_list_line_set, iou_2d_threshold=0.3
    ):
        if isinstance(xywhs, torch.Tensor):
            xywhs = xywhs.cpu().numpy()
        if len(xywhs) == 0 or len(list_points_2d) == 0:
            return []

        # compute iou_2d_matrix
        iou_2d_matrix = np.zeros((len(xywhs), len(list_points_2d)))
        for i in range(len(xywhs)):
            for j in range(len(list_points_2d)):
                iou_2d_matrix[i, j] = compute_2d_iou(
                    np.hstack([xywhs[i], np.array(0.0)]),
                    np.hstack(
                        [self._get_largest_2d_bbox(list_points_2d[j]), np.array(0.0)]
                    ),
                )
        # fitler valid 3d detection
        valid_idx = []
        for n in range(len(iou_2d_matrix)):
            if iou_2d_matrix[n].max() < iou_2d_threshold:
                valid_idx.append(valid_list_line_set[iou_2d_matrix[n].argmax()])

        return valid_idx

    def _get_largest_2d_bbox(self, bbox_3d_on_2d: np.ndarray):
        """
        bbox_3d_on_2d: shape(8, 2)

        return [x, y, w, h]
        """
        x_min, x_max, y_min, y_max = (
            bbox_3d_on_2d[:, 0].min(),
            bbox_3d_on_2d[:, 0].max(),
            bbox_3d_on_2d[:, 1].min(),
            bbox_3d_on_2d[:, 1].max(),
        )
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min
        return np.array([x, y, w, h])
