import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io, os
from PIL import Image
import imageio

from sensor_fusion.utils_sf.kalman_filter import Trackers
from utils_sf.utils import transform_3d_detection_to_world_frame
from utils_sf.put_imgs_on_same_figure import Visualizer4Results
class TrackerVisualizer():
    def __init__(self, scene_idx: int, save_dir: str):

        self.scene_idx = scene_idx
        self.save_dir = save_dir

    def visualize(self, tracker, lidar_idx: int, Transformations=None):
        fig, ax = plt.subplots()
        if isinstance(Transformations, np.ndarray):
            ax.set_xlim(-80, 80)
            ax.set_ylim(-80, 80)
        else:
            ax.set_xlim(-200, 200)
            ax.set_ylim(-200, 200)

        for obj in tracker.tracked_objs:
            if isinstance(Transformations, np.ndarray):
                transformed_state = transform_3d_detection_to_world_frame(obj.state[[0, 1, 2, 6, 7, 8, 9]][None, :] ,Transformations)
                self._draw_rectangle(ax, (transformed_state[0, 0], transformed_state[0, 1]), obj.state[6], obj.state[7], np.degrees(transformed_state[0, 6]))
            else:
                self._draw_rectangle(ax, (obj.state[0], obj.state[1]), obj.state[6], obj.state[7], np.degrees(obj.state[9]))

        self._draw_rectangle(ax, (0, 0), 4.2, 1.7, np.degrees(0), linewidth=2, edgecolor='g', facecolor='g')

        lidar_dir_path = os.path.join(
            self.save_dir, os.path.join(
                'idx_scene_' + str(self.scene_idx), 'idx_lidar_' + str(lidar_idx)
                )
            )
        det_2d_vis_path = os.path.join(lidar_dir_path, 'det_2d_vis_unrotated.png')
        ax.axis('off')
        os.makedirs(lidar_dir_path, exist_ok=True)
        plt.savefig(det_2d_vis_path, dpi=200)
        plt.close()

    def _draw_rectangle(self, ax, center, width, height, angle, linewidth=2, edgecolor='r', facecolor='none'):
        ### center: [x, y], width, height, angle: degree ###
        rectangle = patches.Rectangle(
            (center[0] - width / 2.0, center[1] - height / 2.0), 
            width, 
            height, 
            angle=angle,
            linewidth=linewidth, 
            edgecolor=edgecolor, 
            facecolor=facecolor
        )
        ax.add_patch(rectangle)

    def generate_anim(self, ani_name='2d_visualization.gif'):
        vis = Visualizer4Results()
        vis.process_sequence_imgs(os.path.join(self.save_dir, 'idx_scene_' + str(self.scene_idx)))
        vis.generate_anim(ani_name)