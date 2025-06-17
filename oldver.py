# flake8: noqa
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.morphology import binary_closing
from scipy.spatial.distance import cdist
from scipy.ndimage import convolve
import shutil
from pathlib import Path
import cv2
from utils import make_axis_attention_map, find_largest_incircle, find_largest_incircle_attention, find_largest_inscribed_circle, merge_skeletons_and_smooth
from pca import find_major_axis_pca, draw_pca_result


palette=[
  0, 0, 0,
  255, 0, 0,
  0, 255, 0,
  0, 0, 255,
  0, 255, 255,
  255, 255, 0,
  255, 0, 255
    
]
palette += [0] * (256*3 - len(palette))
    
    
# 필요한 3개의 클래스에 대한 스켈레톤만 시각화
image_dir = "/mnt/home/chaelin/hyunjung/skeleton/data/train/labels"
output_dir = r"/mnt/home/chaelin/hyunjung/skeleton/sk/allSk"
os.makedirs(output_dir, exist_ok=True)
images = os.listdir(image_dir)


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
    return endpoints


#yx xy 바꾸기
def yx2xy(pt):
    return (pt[1],pt[0])


#직선 연장해서 > 마스크값이 1에서 0으로 바뀌는 지점 찾기
def extend_line_find_transition(mask, start_point,end_point,direction='end',num_points=100):
    """
    start, end : y,x
    direction: start/end 연장방향
    num_points: 연장선 위에서 샘플링할 점 개수
    """
    start_point,end_point=np.array(start_point),np.array(end_point)
    if direction=='end':
        vector=end_point-start_point
        base_point=end_point
    elif direction=='start':
        vector=start_point-end_point
        base_point=start_point
    else:
        raise ValueError("direction은 start나 end만 가능합니다")
    
    unit_vector=vector/np.linalg.norm(vector)
    extend_length=50
    extended_point=base_point+unit_vector*extend_length
    line_points=np.linspace(base_point,extended_point,num_points)
    
    #마스크값 변화 탐색
    prev_value=1
    for pt in line_points:
        y,x = int(round(pt[0])),int(round(pt[1]))
        if y<0 or y>=mask.shape[0] or x<0 or x>=mask.shape[1]:
            break
        current_value=mask[y,x]
        if prev_value==1 and current_value==0:
            return (y,x)
        prev_value=current_value
    return None


#여러개의 점이 들어오면 가장 멀리 떨어진 두 점만 반환
def pick_two_farthest_points(coords):
    if len(coords) <=2 :
        return coords
    dists = cdist(coords,coords)
    i,j=np.unravel_index(np.argmax(dists), dists.shape)
    return np.array([coords[i],coords[j]])
    


for image_name in images:
    image_path = os.path.join(image_dir, image_name)
    img = np.array(Image.open(image_path))
    all_skeletons = np.zeros_like(img)
    print(f"Starting image process of {image_path}")

    class_ids = [2, 3, 4]  # endo 2, cervix 3, fundus 4 클래스 번호
    skeletons = {}
    endpoints_dict={}

    for class_id in class_ids:
        mask = (img == class_id).astype(np.uint8)
        mask = binary_closing(mask, footprint=np.ones((3, 3)))
        skeleton = skeletonize(mask)
        skeletons[class_id] = skeleton
        all_skeletons[skeleton > 0] = class_id
        endpoints_dict[class_id]=find_endpoints(skeleton)
        
    endo_skeleton = skeletons[2]
    cervix_skeleton = skeletons[3]
    fundus_skeleton = skeletons[4]

    #cervix 스켈레톤 너무 짧은 경우
    if len(np.argwhere(endpoints_dict[3]))==1:
        cervix_start=tuple(np.argwhere(endpoints_dict[3]))[0]
    else:
        cervix_start, cervix_end=tuple(np.argwhere(endpoints_dict[3]))[0] , tuple(np.argwhere(endpoints_dict[3]))[1]
    #fundus 스켈레톤 너무 짧은 경우
    if len(np.argwhere(endpoints_dict[4]))==1:
        fundus_start=tuple(np.argwhere(endpoints_dict[4]))[0]
    else:
        fundus_start, fundus_end=tuple(np.argwhere(endpoints_dict[4]))[0] , tuple(np.argwhere(endpoints_dict[4]))[1]

    img_pil = Image.fromarray(img.astype(np.uint8), mode="P")
    img_pil.putpalette(palette)
    img_pil=img_pil.convert('RGB')
    
    combined_mask=np.isin(img,[2,3,4]).astype(np.uint8)
    combined_mask=binary_closing(combined_mask,footprint=np.ones((3,3)))
    smoothed_points, combined_skeleton = merge_skeletons_and_smooth(combined_mask, smoothing_sigma=2)
    
    skeleton=skeletonize(combined_mask)
    out_img = Image.fromarray(all_skeletons, mode="P")
    out_img.putpalette(palette)
    out_img=np.array(out_img.convert('RGB'))
    
    #선 하나로 긋기 
    mask_fundus = (img == 4).astype(np.uint8)
    mask_cervix = (img == 3).astype(np.uint8)
    mask_endo = (img == 2).astype(np.uint8)
    mask_uterus = fill_mask(img).astype(np.uint8)
    transition_fundus = extend_line_find_transition(mask_fundus, fundus_start, fundus_end, direction='end')
    transition_cervix = extend_line_find_transition(mask_cervix,cervix_start, cervix_end, direction='start')
    print(f"transition_fundus:{transition_fundus}, transition_cervix:{transition_cervix}")
    
    end1_2 = tuple(cervix_skeleton[-1])  # cervix_skeleton의 마지막 점
    end2_1 = tuple(endo_skeleton[0])     # endo_skeleton의 첫 점
    end2_2 = tuple(endo_skeleton[-1])
    end3_1 = tuple(fundus_skeleton[0])

    full_line = []
    full_line.extend(cervix_skeleton)
    full_line.extend([end1_2, end2_1])
    full_line.extend(endo_skeleton)
    full_line.extend([end2_2, end3_1])
    full_line.extend(fundus_skeleton)
    extended_line = [transition_cervix] + full_line + [transition_fundus]
    for i in range(len(extended_line) - 1):
        cv2.line(out_img, extended_line[i], extended_line[i+1], (255,0,0),1)
        
    # cv2.line(out_img,yx2xy(fundus_start),yx2xy(transition_fundus),(255,0,0),1)
    # cv2.line(out_img,yx2xy(transition_cervix),yx2xy(cervix_end),(255,0,0),1)
    
    #길이 재기
    length_fundus=calc_distance(fundus_start,transition_fundus)
    length_cervix=calc_distance(transition_cervix,cervix_end)
    
    #최대 내접원 찾기
    (endo_center_x, endo_center_y), endo_diameter = find_largest_incircle(mask_endo,image_name,save_dir="endo_circle")
    (uterus_center_x, uterus_center_y), uterus_diameter = find_largest_incircle(mask_uterus,image_name,save_dir="uterus_circle")
    (attn_endo_center_x, attn_endo_center_y), attn_endo_diameter = find_largest_incircle_attention(mask_endo,image_name,save_dir="attn_endo_circle")
    (attn_uterus_center_x, attn_uterus_center_y), attn_uterus_diameter = find_largest_incircle_attention(mask_uterus,image_name,save_dir="attn_uterus_circle")
    result = find_largest_inscribed_circle(mask_endo, endo_skeleton)
    uterus_result = find_largest_inscribed_circle(mask_uterus, combined_skeleton)

    
    results=[
        f"{image_name} results",
        "attention 없이: ",
        f"최대 내막 내접원의 중심: ({endo_center_x}, {endo_center_y}), 지름: {endo_diameter}",
        f"최대 uterus 내접원의 중심: ({uterus_center_x}, {uterus_center_y}), 지름: {uterus_diameter}",
        f"스켈레톤 따라가는 최대 내접원 중심: ({result[0]}, {result[1]}), 지름: {result[2]}",
        f"스켈레톤 따라가는 유터러스 내접원 중심: ({uterus_result[0]}, {uterus_result[1]}), 지름: {uterus_result[2]}",
        "attention 있이:"
        f"최대 내막 내접원의 중심: ({attn_endo_center_x}, {attn_endo_center_y}), 지름: {attn_endo_diameter}",
        f"최대 uterus 내접원의 중심: ({attn_uterus_center_x}, {attn_uterus_center_y}), 지름: {attn_uterus_diameter}"

    ]

    
    #엔도 스켈레톤 이진화> 컨투어 추출> 길이계산
    endo_skeleton=(endo_skeleton>0).astype(np.uint8)
    contours,_=cv2.findContours(endo_skeleton,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if contours:
        main_contour=max(contours,key=lambda x:cv2.arcLength(x, False))
        length_endo=cv2.arcLength(main_contour,closed=False)
    else:
        length_endo=0

    
    # 시각화 및 저장
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img_pil)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    ax[1].imshow(out_img)
    ax[1].axis('off')
    ax[1].set_title('Combined skeleton', fontsize=20)
    ax[1].scatter([transition_cervix[1], transition_fundus[1]], [transition_cervix[0], transition_fundus[0]], c='red', s=5)
    
    # 각 클래스별 endpoint에 점 찍기
    colors = {2: 'lime', 3: 'blue', 4: 'skyblue'}
    for class_id in [2, 3, 4]:
        endpoints = find_endpoints(skeletons[class_id])
        coords = np.argwhere(endpoints)
        if coords.size > 2:
            coords=pick_two_farthest_points(coords)
        ys, xs = coords[:, 0], coords[:, 1]
        ax[1].scatter(xs, ys, s=5, facecolors=colors[class_id])

    ax[1].legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    base_name = os.path.splitext(image_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}_allSk.png")
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Saved skeleton image for {image_name} at {save_path}")
    
    with open('results.txt','a') as f:
        for line in results:
            f.write(line+'\n')
    





