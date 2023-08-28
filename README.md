# Sensor Fusion of Lidar and Cameras

This repository provides code and guidelines for sensor fusion of lidar and cameras. Below are the steps for installation and bug fixes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [OpenPCDet](#openpcdet)
  - [Other Dependencies](#other-dependencies)
- [Bug Fixes](#bug-fixes)
  - [Bug related to Argoverse2 in Openpcdet](#bug-related-to-argoverse2-in-openpcdet)
  - [Invalid Official Checkpoint of Openpcdet](#invalid-official-checkpoint-of-openpcdet)
- [Download Dataset](#download-dataset)
- [Get Started](#get-started)

## Prerequisites

- Python 3.8/3.9/3.10
- Follow [official documentation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md).
- Notice: The same version of local cuda and corresponding pytorch-cuda is mandatory for the installation of Openpcdet.

## Installation

### OpenPCDet

First, install OpenPCDet following the guidelines provided in their [official documentation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md).

### Other Dependencies

After installing OpenPCDet, install other required Python packages by running:

```bash
pip install open3d==0.17.0 kornia==0.5.8 opencv-python==4.8.0.76 av2
```

## Bug Fixes

### Bug related to Argoverse2 in Openpcdet

To resolve a bug related to inference data from Argoverse2, you'll need to manually update the \`box_colormap\` in \`tools/visual_utils/open3d_vis_utils.py\` to the following:

```python
box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0.5, 0.5, 0.5],
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5],
    [0.5, 0.5, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0.5],
    [0.7, 0.3, 0.3],
    [0.8, 0.6, 0.3],
    [0.2, 0.8, 0.5],
    [0.6, 0.2, 0.7],
    [0.3, 0.7, 0.8],
    [0.8, 0.3, 0.2],
    [0.3, 0.2, 0.8],
    [0.8, 0.5, 0.6],
    [0.4, 0.7, 0.2],
    [0.7, 0.2, 0.4],
    [0.2, 0.4, 0.7],
    [0.6, 0.3, 0.5],
    [0.5, 0.6, 0.3],
    [0.4, 0.2, 0.1],
    [0.1, 0.4, 0.2],
    [0.2, 0.1, 0.4]
]
```


### Invalid Official Checkpoint of Openpcdet

Please note that the pretrained Argoverse2 model in the official repository has issues. Use the one provided in the \`tools/argo2_model\` directory instead.

## Download Dataset
Download dataset in this link [Argoverse2](https://www.argoverse.org/av2.html#sensor-link).
The root_path has to be arranged like this (just the default format):
```
- root_path
    - scene_ids (e.g. 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a)
        - calibration
            - egovehicle_SE3_sensor.feather
            - intrinsics.feather
        - map
            ...
        - sensors
            - cameras
                - ring_front_center
                    - xxx.jpg
                    ...
                - ring_front_left
                - ring_front_right
                ...
                - stereo_front_left
                - stereo_front_right
            - lidar
                - xxx.feather
        - annotations.feather
        - city_SE3_egovehicle.feather
```

## Get Started
Get started by running:
```python
python sensor_fusion/main.py --root_path YOUR/DATASET/PATH
```