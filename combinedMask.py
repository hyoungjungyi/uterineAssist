# flake8: noqa
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.morphology import binary_closing
from scipy.spatial.distance import cdist
from scipy.ndimage import convolve, distance_transform_edt
import shutil
from pathlib import Path
import cv2

def extend_line_find_transition(
    mask,
    start_point,
    end_point,
    transition_type='fundus',  # 'fundus' 또는 'cervix'
    num_points=100
):
    """
    start, end : (y, x) 좌표
    transition_type: 
        - 'fundus' → 2→3 전환 탐색 (start 방향 확장)
        - 'cervix' → 2→4 전환 탐색 (end 방향 확장)
    """
    # 타겟 및 방향 설정
    if transition_type == 'fundus':
        target_from, target_to = 2, 3
        direction = 'start'
    elif transition_type == 'cervix':
        target_from, target_to = 2, 4
        direction = 'end'
    else:
        raise ValueError("transition_type은 'fundus' 또는 'cervix'만 가능")

    print(f"[DEBUG] transition_type: {transition_type}, direction: {direction}, target_from: {target_from}, target_to: {target_to}")

    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    if direction == 'end':
        vector = end_point - start_point
        base_point = end_point
    else:
        vector = start_point - end_point
        base_point = start_point
    
    unit_vector = vector / np.linalg.norm(vector)
    extend_length = 50
    extended_point = base_point + unit_vector * extend_length
    line_points = np.linspace(base_point, extended_point, num_points)
    
    prev_value = None
    for idx, pt in enumerate(line_points):
        y, x = int(round(pt[0])), int(round(pt[1]))
        if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
            print(f"[DEBUG] Out of bounds at ({y}, {x}), break.")
            break
        
        current_value = mask[y, x]
        print(f"[DEBUG] 1st scan idx:{idx} ({y},{x}) value={current_value}")

        if prev_value == target_from and current_value == target_to:
            print(f"[DEBUG] Transition {target_from}->{target_to} found at ({y},{x})")
            # 1 찾을 때까지 추가 확장
            step_size = 1
            current_pos = np.array([y, x], dtype=float)
            for step in range(extend_length * 2):
                current_pos += unit_vector * step_size
                y_new, x_new = int(round(current_pos[0])), int(round(current_pos[1]))
                if y_new < 0 or y_new >= mask.shape[0] or x_new < 0 or x_new >= mask.shape[1]:
                    print(f"[DEBUG] 2nd scan out of bounds at ({y_new}, {x_new}), break.")
                    break
                print(f"[DEBUG] 2nd scan step:{step} ({y_new},{x_new}) value={mask[y_new, x_new]}")
                if mask[y_new, x_new] == 1:
                    print(f"[DEBUG] 1 found at ({y_new},{x_new}) in 2nd scan")
                    return (y_new, x_new)
            print("[DEBUG] No 1 found after transition.")
            return None
        
        prev_value = current_value
    
    print("[DEBUG] No target_from → target_to transition found.")
    return None
