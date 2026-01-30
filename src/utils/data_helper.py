#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
Data preprocessing helper functions for Sydney Urban Objects Dataset / KITTI.
Includes point cloud loading, voxelization, and augmentation.
"""

import numpy as np
import sys
import os
import glob

# Sydney Urban Objects LiDAR dataset formats
FIELDS = ['t', 'intensity', 'id', 'x', 'y', 'z', 'azimuth', 'range', 'pid']
TYPES = ['int64', 'uint8', 'uint8', 'float32', 'float32', 'float32', 'float32', 'float32', 'int32']

# Label dictionary
SUOD_LABEL_MAP = {
    '4wd': 0, 'building': 1, 'bus': 2, 'car': 3, 'pedestrian': 4, 'pillar': 5, 'pole': 6,
    'traffic_lights': 7, 'traffic_sign': 8, 'tree': 9, 'truck': 10, 'trunk': 11, 'ute': 12, 'van': 13
}
SUOD_LABEL_MAP_REV = {v: k for k, v in SUOD_LABEL_MAP.items()}

OCCUPIED = 1
FREE = 0

def load_points_from_bin(bin_file, with_intensity=False):
    """
    Load point cloud from a binary file.
    """
    bin_type = np.dtype(dict(names=FIELDS, formats=TYPES))
    data = np.fromfile(bin_file, bin_type)

    # Stack to (N, 3) or (N, 4)
    if with_intensity:
        points = np.vstack([data['x'], data['y'], data['z'], data['intensity']]).T
    else:
        points = np.vstack([data['x'], data['y'], data['z']]).T

    return points


def get_SUOD_label(index):
    """Retrieve string label from integer index."""
    return SUOD_LABEL_MAP_REV.get(index, "Unknown")


def save_pcd_from_bin(bin_file, with_intensity=False):
    """
    Read bin file and convert to ASCII .pcd format without using external PCL library.
    Compatible with Python 3.5 (No f-strings).
    """
    points = load_points_from_bin(bin_file, with_intensity)
    out_file = bin_file.replace('.bin', '.pcd')
    
    num_points = points.shape[0]
    
    # Python 3.5 compatible formatting
    headers = [
        '# .PCD v0.7 - Point Cloud Data file format',
        'VERSION 0.7',
        'FIELDS x y z{}'.format(" intensity" if with_intensity else ""),
        'SIZE 4 4 4{}'.format(" 4" if with_intensity else ""),
        'TYPE F F F{}'.format(" F" if with_intensity else ""),
        'COUNT 1 1 1{}'.format(" 1" if with_intensity else ""),
        'WIDTH {}'.format(num_points),
        'HEIGHT 1',
        'VIEWPOINT 0 0 0 1 0 0 0',
        'POINTS {}'.format(num_points),
        'DATA ascii'
    ]
    
    with open(out_file, 'w') as f:
        f.write('\n'.join(headers) + '\n')
        np.savetxt(f, points, fmt='%.4f')
    
    print("Saved: {}".format(out_file))


def voxelize(points, voxel_size=(24, 24, 24), padding_size=(32, 32, 32), resolution=0.1):
    """
    Convert `points` to centralized voxel grid.
    
    Args:
        points: (N, 3) numpy array
        voxel_size: Dimensions of the object content (logical size)
        padding_size: Dimensions of the output tensor (padded size)
        resolution: Voxel resolution in meters
        
    Returns:
        voxels: 3D Occupancy Grid (padded size)
        inside_box_points: The subset of points that fit in the box
    """
    if abs(resolution) < sys.float_info.epsilon:
        print('Error: Resolution cannot be zero')
        return None, None

    # Remove NaNs
    points = points[~np.isnan(points).any(axis=1)]

    if points.shape[0] == 0:
        return np.zeros(padding_size), points

    # Normalize to origin (0,0,0) based on min values
    origin = np.min(points, axis=0)
    points = points - origin

    # Filter points outside the logical voxel_size
    limit = np.array(voxel_size) * resolution
    mask = np.all((points >= 0) & (points < limit), axis=1)
    inside_box_points = points[mask]

    # Initialize voxel grid
    voxels = np.zeros(padding_size, dtype=np.int8)
    
    # Centralize: Offset points to center them within the padding box
    offset_grids = (np.array(padding_size) - np.array(voxel_size)) / 2.0
    offset_meters = offset_grids * resolution
    
    center_points = inside_box_points + offset_meters

    # Discretize to indices
    indices = (center_points / resolution).astype(int)

    # Clip indices to ensure safety
    indices[:, 0] = np.clip(indices[:, 0], 0, padding_size[0] - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, padding_size[1] - 1)
    indices[:, 2] = np.clip(indices[:, 2], 0, padding_size[2] - 1)

    # Set occupancy
    voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = OCCUPIED

    return voxels, inside_box_points


def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    """
    Apply rotation (rx,ry,rz) first, then translation (tx,ty,tz).
    """
    # Create Rotation Matrices
    # Rx
    c, s = np.cos(rx), np.sin(rx)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    
    # Ry
    c, s = np.cos(ry), np.sin(ry)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    # Rz
    c, s = np.cos(rz), np.sin(rz)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # Combined Rotation R = Rz * Ry * Rx (standard Euler order)
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Apply Rotation: P_rot = P @ R.T (because points are row vectors)
    points_rot = np.dot(points, R.T)

    # Apply Translation
    return points_rot + np.array([tx, ty, tz])


def aug_data(points, aug_size, uniform_rotate_only=False):
    """
    Data augmentation via rotation and translation.
    """
    rot_interval = 2 * np.pi / (aug_size + 1)
    points_list = [points]

    for idx in range(1, aug_size + 1):
        if uniform_rotate_only:
            r_z = rot_interval * idx
            t_x = t_y = t_z = 0.0
        else:
            # Random jitter
            r_z = np.random.uniform(-np.pi / 10, np.pi / 10)
            t_x = np.random.normal(0, 0.1) 
            t_y = np.random.normal(0, 0.1)
            t_z = np.random.normal(0, 0.05)

        points_aug = point_transform(points, t_x, t_y, t_z, rz=r_z)
        points_list.append(points_aug)

    return np.array(points_list, dtype=np.float32)


def load_data_from_npy(npy_dir, mode='training'):
    """
    Load voxelized data and labels from .npy files.
    """
    input_path = os.path.join(npy_dir, mode, '*.npy')
    
    voxels_list = []
    label_list = []
    
    files = glob.glob(input_path)
    if not files:
        print("Warning: No files found in {}".format(input_path))
        return np.array([]), np.array([])

    for npy_f in files:
        # Parse label from filename: e.g. "pillar.2.3582_12.npy" -> "pillar"
        filename = os.path.basename(npy_f)
        class_name = filename.split('.')[0]
        
        if class_name in SUOD_LABEL_MAP:
            label = SUOD_LABEL_MAP[class_name]
            label_list.append(label)
            
            voxels = np.load(npy_f).astype(np.float32)
            voxels_list.append(voxels)
        else:
            print("Skipping unknown class: {}".format(class_name))

    return np.array(voxels_list), np.array(label_list)