import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
from PIL import Image
import imageio

from sensor_fusion.utils_sf.kalman_filter import Trackers

class TrackerVisualizer():
    def __init__(self):
        self.buffer_list = []

    def visualize(self, tracker):
        fig, ax = plt.subplots()

        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)

        for obj in tracker.tracked_objs:
            self._draw_rectangle(ax, (obj.state[0], obj.state[1]), obj.state[6], obj.state[7], np.degrees(obj.state[9]))
            
        # plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        self.buffer_list.append(buf)
        plt.close(fig)

    def _draw_rectangle(self, ax, center, width, height, angle):
        ### center: [x, y], width, height, angle: degree ###
        rectangle = patches.Rectangle(
            (center[0] - width / 2.0, center[1] - height / 2.0), 
            width, 
            height, 
            angle=angle,
            linewidth=1, 
            edgecolor='r', 
            facecolor='none'
        )
        ax.add_patch(rectangle)

    def generate_anim(self, ani_name='2d_visualization.gif'):
        img_arrays = []
        for buf in self.buffer_list:
            img_array = np.array(Image.open(buf))
            img_arrays.append(img_array)
            buf.close()
        with imageio.get_writer(ani_name, mode='I', duration=0.05) as writer:
            for image in img_arrays:
                writer.append_data(image)