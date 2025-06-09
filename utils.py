# flake8: noqa
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.morphology import binary_closing , skeletonize
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt, binary_fill_holes, gaussian_filter1d
import shutil
from pathlib import Path
import cv2

def make_axis_attention_map(mask):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt=max(contours,key=cv2.contourArea)
    (cx,cy),(width,height),angle=cv2.fitEllipse(cnt)
    #장축 방향 단위 벡터
    theta=np.deg2rad(angle)
    dx,dy=np.cos(theta),np.sin(theta)
    H,W=mask.shape
    Y,X=np.meshgrid(np.arange(H),np.arange(W),indexing='ij')
    vec_x=X-cx
    vec_y=Y-cy
    #장축 방향으로 투영
    axis_proj=vec_x*dx+vec_y*dy
    #장축 방향에 가까울수록 값이 높
    attention=np.exp(-(axis_proj**2)/(0.2*width)**2)
    attention=attention*(mask>0)
    return attention/attention.max()
    


#마스크에 내접하는 가장 큰 원 찾기 - with attention
def find_largest_incircle_attention(mask, image_name="default", save_dir=None):
    binary_mask = (mask > 0).astype(bool)
    filled_mask = binary_fill_holes(binary_mask).astype(np.uint8) * 255
    dist_map = distance_transform_edt(filled_mask)
    attention = make_axis_attention_map(filled_mask)
    attn_dist_map = dist_map * attention

    dist_map = distance_transform_edt(filled_mask)
    if np.max(dist_map) == 0:
        return (0, 0), 0, filled_mask
    
    max_radius = np.max(dist_map)
    max_row, max_col = np.unravel_index(np.argmax(attn_dist_map), attn_dist_map.shape)
    center = (int(max_col), int(max_row))
    diameter = int(2 * dist_map[max_row,max_col])
    
    #(마스크: 파란색, 내접원: 초록색)
    save_path = os.path.join(save_dir, f"{image_name}.png")
    os.makedirs(save_dir, exist_ok=True)
    
    result_img = np.zeros((*filled_mask.shape, 3), dtype=np.uint8)
    result_img[filled_mask == 255] = [255, 255, 255]  # BGR 포맷
    cv2.circle(
        result_img,
        center,
        int(dist_map[max_row,max_col]),
        (0,0,255),
        2
    )
    
    cv2.imwrite(save_path, result_img)
    
    return center, diameter


#마스크에 내접하는 가장 큰 원 찾기 - w/o attention
def find_largest_incircle(mask, image_name="default", save_dir=None):
    binary_mask = (mask > 0).astype(bool)
    filled_mask = binary_fill_holes(binary_mask).astype(np.uint8) * 255


    dist_map = distance_transform_edt(filled_mask)
    if np.max(dist_map) == 0:
        return (0, 0), 0, filled_mask
    
    max_radius = np.max(dist_map)
    max_row, max_col = np.unravel_index(np.argmax(dist_map), dist_map.shape)
    center = (int(max_col), int(max_row))
    diameter = int(2 * max_radius)
    
    #(마스크: 파란색, 내접원: 초록색)
    save_path = os.path.join(save_dir, f"{image_name}.png")
    os.makedirs(save_dir, exist_ok=True)
    
    result_img = np.zeros((*filled_mask.shape, 3), dtype=np.uint8)
    result_img[filled_mask == 255] = [255, 255, 255]  # BGR 포맷
    cv2.circle(
        result_img,
        center,
        int(dist_map[max_row,max_col]),
        (0,0,255),
        2
    )
    
    cv2.imwrite(save_path, result_img)
    
    return center, diameter


#스켈레톤 따라가기
def find_largest_inscribed_circle(mask, skeleton):

    mask_bin = (mask > 0).astype(np.uint8)
    dist_map = distance_transform_edt(mask_bin)
    skel_points = np.column_stack(np.where(skeleton > 0))  # (y, x) 형식
    max_radius = -1
    best_center = (0, 0)
    
    for y, x in skel_points:
        current_radius = dist_map[y, x]
        if current_radius > max_radius:
            max_radius = current_radius
            best_center = (x, y)  # OpenCV 좌표 체계(x,y)
    
    return (*best_center, 2 * max_radius)



def merge_skeletons_and_smooth(mask, smoothing_sigma=2):
    binary_mask=(mask>0).astype(np.uint8)
    closed_mask=binary_closing(binary_mask,footprint=np.ones((3,3)))
    skeleton=skeletonize(closed_mask).astype(np.uint8)
    skel_points=np.column_stack(np.where(skeleton>0))
    if len(skel_points)==0:
        return np.array([]), skeleton
    #greedy로 가장 가까운 점끼리 연결
    points=skel_points.copy()
    ordered=[points[0]]
    points=np.delete(points,0,axis=0)
    while len(points)>0:
        last=ordered[-1]
        dists=cdist([last],points)
        idx=np.argmin(dists)
        ordered.append(points[idx])
        points=np.delete(points,idx,axis=0)
    ordered=np.array(ordered)
    y_smooth=gaussian_filter1d(ordered[:,0],sigma=smoothing_sigma)
    x_smooth=gaussian_filter1d(ordered[:,1],sigma=smoothing_sigma)
    smoothed_points=np.stack([y_smooth,x_smooth],axis=1)
    
    return smoothed_points, skeleton
    