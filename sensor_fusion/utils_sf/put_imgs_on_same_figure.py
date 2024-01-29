import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os, io
import imageio
import numpy as np
from tqdm import tqdm


class Visualizer4Results():
    def __init__(self):
        self.cam_names_to_add = ['ring_front_center',
                                 'ring_front_left',
                                 'ring_front_right',
                                 'ring_side_left',
                                 'ring_side_right',
                                 'ring_rear_left',
                                 'ring_rear_right',]

        self.center_img_name = 'det_2d_vis_unrotated.png'
        self.buffer_list = []

    def _put_imgs_on_one_figure(self, dir_path):
        list_img_paths = {}
        for cam_name in self.cam_names_to_add:
            img_file_name = os.listdir(os.path.join(dir_path, cam_name))[0]
            img_path = os.path.join(os.path.join(dir_path, cam_name), img_file_name)
            list_img_paths[cam_name] = img_path

        fig = plt.figure()
        gs = gridspec.GridSpec(3, 3,
                       width_ratios=[1, 4, 1],
                       height_ratios=[1, 4, 1]
                       )

        # fig, axes = plt.subplots(3, 3)

        # for ax in axes.ravel():
        #     ax.axis('off')

        ring_front_center = Image.open(list_img_paths['ring_front_center'])
        ring_front_left = Image.open(list_img_paths['ring_front_left'])
        ring_front_right = Image.open(list_img_paths['ring_front_right'])
        ring_rear_left = Image.open(list_img_paths['ring_rear_left'])
        ring_rear_right = Image.open(list_img_paths['ring_rear_right'])
        ring_side_left = Image.open(list_img_paths['ring_side_left'])
        ring_side_right = Image.open(list_img_paths['ring_side_right'])

        center_img = Image.open(os.path.join(dir_path, self.center_img_name)).rotate(90, expand=True)

        # axes[0, 1].imshow(ring_front_center)
        # axes[0, 0].imshow(ring_front_left)
        # axes[0, 2].imshow(ring_front_right)
        # axes[1, 1].imshow(center_img)
        # axes[1, 0].imshow(ring_side_left)
        # axes[1, 2].imshow(ring_side_right)
        # axes[2, 0].imshow(ring_rear_left)
        # axes[2, 2].imshow(ring_rear_right)


        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[0, 2])
        ax4 = plt.subplot(gs[1, 0])
        ax5 = plt.subplot(gs[1, 1])
        ax6 = plt.subplot(gs[1, 2])
        ax7 = plt.subplot(gs[2, 0])
        ax8 = plt.subplot(gs[2, 2])

        ax1.imshow(ring_front_left)
        ax2.imshow(ring_front_center)
        ax3.imshow(ring_front_right)
        ax4.imshow(ring_side_left)
        ax5.imshow(center_img)
        ax6.imshow(ring_side_right)
        ax7.imshow(ring_rear_left)
        ax8.imshow(ring_rear_right)

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
            ax.axis('off')


        plt.savefig('temp.png', dpi=500)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=400)
        buf.seek(0)
        self.buffer_list.append(buf)
        plt.close(fig)

    def process_sequence_imgs(self, dir_path):
        list_dir_names = os.listdir(dir_path)
        def sort_key(s):
            return int(s.split('_')[-1])
        list_dir_names = sorted(list_dir_names, key=sort_key)
        for dir_name in tqdm(list_dir_names):
            self._put_imgs_on_one_figure(os.path.join(dir_path, dir_name))

    def generate_anim(self, ani_name='2d_visualization.mp4'):
        img_arrays = []
        for buf in self.buffer_list:
            img_array = np.array(Image.open(buf))
            img_arrays.append(img_array[:, :, :3])
            buf.close()
        # with imageio.get_writer(ani_name, mode='I', fps=20, codec='libx264') as writer:
        #     for image in img_arrays:
        #         writer.append_data(image)
        # writer.close()
        np_img = np.stack(img_arrays, axis=0)
        # imageio.mimwrite(ani_name, np_img, fps=10, codec='libx264')
        imageio.mimwrite(ani_name, np_img, duration=100, codec='libx264')

if __name__ == '__main__':
    vis = Visualizer4Results()
    vis.process_sequence_imgs('idx_scene_0')
    vis.generate_anim('2d_visualization_0_with_vel_obs_new_0pt8.mp4')