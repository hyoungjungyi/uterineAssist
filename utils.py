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
from scipy.ndimage import convolve
from skimage.draw import line



#마스크에 내접하는 가장 큰 원 찾기
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
    

def fill_holes_opencv(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    im_floodfill = mask.copy()
    h, w = mask.shape[:2]
    mask2 = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask2, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    filled = mask | im_floodfill_inv
    return filled

def draw_major_axis(mask):
    filled = fill_holes_opencv(mask)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)
    cnt = max(contours, key=cv2.contourArea)
    (x, y), (width, height), angle = cv2.fitEllipse(cnt)
    major_axis = max(width, height)
    radius = int(major_axis / 2)
    color_img = cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_img, (int(x), int(y)), radius, (0, 0, 255), 1)  # 빨강 원
    return color_img

def fill_mask(mask):

    mask = (mask > 0).astype(np.uint8) * 255
    mask=np.where(mask>1,1,mask)

    return mask



def calc_distance(pt1,pt2):
    pt1=np.array(pt1)
    pt2=np.array(pt2)
    return np.linalg.norm(pt1-pt2)


def find_endpoints(skeleton):
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    filtered = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    # 중앙(10) + 이웃 1개(1) = 11인 곳이 endpoint
    endpoints = (filtered == 11)
    if np.sum(skeleton)==1:
        endpoints |= skeleton.astype(bool)
    y,x=np.where(endpoints)
    endpoint_coords=list(zip(y,x))
    return endpoint_coords


#yx xy 바꾸기
def yx2xy(pt):
    return (pt[1],pt[0])

def convert_to_int_tuple(point):
    if isinstance(point, tuple):
        return (int(point[0]), int(point[1]))
    else:  # numpy array 등
        return (int(point[0]), int(point[1]))


#방향으로 1->0 경계점 찾기
def extend_to_boundary(mask,point,direction,max_length=500):
    point=np.array(point)
    direction=np.array(direction)
    unit_vector=direction/np.linalg.norm(direction)
    for step in range(1,max_length):
        extended_point=point+unit_vector*step
        y,x=int(round(extended_point[0])),int(round(extended_point[1]))
        if y<0 or y>=mask.shape[0] or x<0 or x>=mask.shape[1]:
            break
        if mask[y,x]==0:
            return(y,x)
    return None

def get_direction_with_slope_and_uterus(start,end, mask_uterus):
    start,end=np.array(start),np.array(end)
    slope_vector = end - start  # start→end 방향 벡터

    ys, xs = np.where(mask_uterus == 1)
    if len(ys) == 0 or len(xs) == 0:
        return None, None, None
    uterus_center = np.array([np.mean(ys), np.mean(xs)])

    # start, end 중 uterus_center에 더 가까운 점 선택
    dist_start = np.linalg.norm(start - uterus_center)
    dist_end = np.linalg.norm(end - uterus_center)
    if dist_start < dist_end:
        # start가 uterus_center에 더 가까우므로 start -> end(즉, slope_vector)가 uterus 방향
        direction = slope_vector
        base_point = end
    else:
        # end가 uterus_center에 더 가까우므로 end -> start(-slope_vector)가 uterus 방향
        direction = -slope_vector
        base_point = start

    # 혹은, start/end에서 uterus_center로 직접 방향 벡터를 구할 수도 있음
    # direction = uterus_center - base_point

    return base_point, direction, uterus_center


    


#여러개의 점이 들어오면 가장 멀리 떨어진 두 점만 반환
def pick_two_farthest_points(coords):
    if len(coords) <=2 :
        return coords
    dists = cdist(coords,coords)
    i,j=np.unravel_index(np.argmax(dists), dists.shape)
    return np.array([coords[i],coords[j]])
    
def get_direction_for_short_skeleton(cervix_point, endo_point, mask_uterus):
    # cervix_point와 endo_point를 numpy array로 변환
    cervix_point = np.array(cervix_point)
    endo_point = np.array(endo_point)

    # cervix에서 endo 방향 (endo_point - cervix_point)
    slope_vector = endo_point - cervix_point

    # cervix에서 endo 반대방향 (cervix_point에서 endo_point 반대방향)
    direction = -slope_vector

    # uterus 중심 계산
    ys, xs = np.where(mask_uterus == 1)
    if len(ys) == 0 or len(xs) == 0:
        return cervix_point, direction, None
    uterus_center = np.array([np.mean(ys), np.mean(xs)])

    return cervix_point, direction, uterus_center

def sort_points_greedy(points, start_idx=0):
    points = np.array(points)
    sorted_points = [points[start_idx]]
    remaining = np.delete(points, start_idx, axis=0)
    for _ in range(len(points)-1):
        last = sorted_points[-1]
        dists = np.sum((remaining - last)**2, axis=1)
        next_idx = np.argmin(dists)
        sorted_points.append(remaining[next_idx])
        remaining = np.delete(remaining, next_idx, axis=0)
    return sorted_points

def connect_points_with_skeleton_check(out_img, skeleton_image, connect_points):
    window = 2  # 5x5 윈도우 (2칸씩 양쪽으로)
    for i in range(len(connect_points)-1):
        pt1 = connect_points[i]    # (y, x)
        pt2 = connect_points[i+1]  # (y, x)
        rr, cc = line(pt1[0], pt1[1], pt2[0], pt2[1])  # (y1, x1, y2, x2)
        rr = np.clip(rr, 0, out_img.shape[0]-1)
        cc = np.clip(cc, 0, out_img.shape[1]-1)
        for r, c in zip(rr, cc):
            r1, r2 = max(0, r-window), min(out_img.shape[0], r+window+1)
            c1, c2 = max(0, c-window), min(out_img.shape[1], c+window+1)
            # 주변 5x5 영역에 스켈레톤이 없으면 직선 그리기
            if np.sum(skeleton_image[r1:r2, c1:c2]) == 0:
                out_img[r, c] = [0, 255, 0]  # 녹색
    return out_img

def custom_connect(out_img, connect_points):
    """
    연결 규칙:
    1. boundary → 4s0 직선 연결
    2. 4s1 → 2s0 직선 연결
    3. 2s1 → 3s0 직선 연결
    """
    # 연결할 포인트 쌍 정의
    connections = [ (0,1), (2,3), (4,5) ]
    six_connections=[ (0,1),(1,2),(3,4) ]
    all_coords=[]
    #cervix의 스켈레톤이 너무 짧아서 end가 1개인경우
    conn = six_connections if len(connect_points) == 6 else connections
    for start_idx, end_idx in conn:
        pt1 = connect_points[start_idx]
        pt2 = connect_points[end_idx]
        rr, cc = line(pt1[0], pt1[1], pt2[0], pt2[1])
        rr = np.clip(rr, 0, out_img.shape[0]-1)
        cc = np.clip(cc, 0, out_img.shape[1]-1)
        out_img[rr, cc] = [0, 255, 0]  # 초록색
        all_coords.extend(list(zip(rr,cc)))
        
    return out_img, all_coords

#각 스켈레톤의 엔드포인트 2개를 비교해서 더 cervix boundary 에 가까운걸 [0]으로, 더 먼걸 [1]로 함
import numpy as np

def reorder_endpoints_by_reference(endpoints_dict, reference_point):
    """
    각 스켈레톤의 양끝점 2개 중 reference_point(cervix_boundary)와의 거리를 계산하여
    더 가까운 점을 [0], 더 먼 점을 [1]로 재정렬하는 함수

    Args:
        endpoints_dict (dict): {class_id: [(y0, x0), (y1, x1)]} 형태의 끝점 딕셔너리
        reference_point (tuple): (y, x) 기준점

    Returns:
        dict: 재정렬된 endpoints_dict
    """
    reordered = {}
    for class_id, points in endpoints_dict.items():
        if len(points) != 2:
            # 끝점이 2개가 아니면 그대로 유지
            reordered[class_id] = points
            continue
        # 각 점과 기준점 사이의 거리 계산
        pt0, pt1 = points
        dist0 = np.linalg.norm(np.array(pt0) - np.array(reference_point))
        dist1 = np.linalg.norm(np.array(pt1) - np.array(reference_point))
        # 더 가까운 점이 [0], 더 먼 점이 [1]
        if dist0 <= dist1:
            reordered[class_id] = [pt0, pt1]
        else:
            reordered[class_id] = [pt1, pt0]
    return reordered

#직교 방향 직선 긋는 함수
def get_orthogonal_direction(pt1, pt2):
    dy = pt2[0] - pt1[0]
    dx = pt2[1] - pt1[1]
    ortho_dy = dx
    ortho_dx = -dy
    length = np.sqrt(ortho_dy**2 + ortho_dx**2)
    if length > 0:
        ortho_dy /= length
        ortho_dx /= length
    return ortho_dy, ortho_dx

#직교하는 선이 2 영역에 얼마나 겹치는지 길이재기
def measure_uterus_thickness(mask, all_coords, out_img, step=2, probe_half_length=30):
    max_thickness = 0
    max_probe_coords = None

    for i in range(0, len(all_coords) - step, step):
        pt1 = all_coords[i]
        pt2 = all_coords[i + step]
        ortho_dy, ortho_dx = get_orthogonal_direction(pt1, pt2)
        center_y = int((pt1[0] + pt2[0]) / 2)
        center_x = int((pt1[1] + pt2[1]) / 2)

        probe_coords = [
            (yy, xx)
            for s in range(-probe_half_length, probe_half_length + 1)
            if 0 <= (yy := int(center_y + s * ortho_dy)) < mask.shape[0]
            and 0 <= (xx := int(center_x + s * ortho_dx)) < mask.shape[1]
        ]

        thickness = sum(mask[y, x] == 1 for y, x in probe_coords)
        if thickness > max_thickness:
            max_thickness = thickness
            max_probe_coords = probe_coords

    if max_probe_coords:
        for y, x in max_probe_coords:
            cv2.circle(out_img, (x, y), 1, (0, 255, 255), -1)  # 노란색 점

    return max_thickness

