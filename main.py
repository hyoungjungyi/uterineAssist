# flake8: noqa
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from PIL import Image
import os
from skimage.morphology import binary_closing
from scipy.spatial.distance import cdist
import shutil
from pathlib import Path
import cv2
from utils import *
import sknw
import networkx as nx
from skimage.draw import line



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
output_dir = r"/mnt/home/chaelin/hyunjung/skeleton/sk/thickness_comparison"
os.makedirs(output_dir, exist_ok=True)
images = os.listdir(image_dir)
skeleton_coords=[]


for image_name in images:
    cervix_end = None
    cervix_point = None
    endo_point = None 
    image_path = os.path.join(image_dir, image_name)
    img = np.array(Image.open(image_path))
    all_skeletons = np.zeros_like(img)
    print(f"Starting image process of {image_path}")

    class_ids = [2, 3, 4]  # endo 2, cervix 4, fundus 3 클래스 번호
    skeletons = {}
    endpoints_dict={}

    for class_id in class_ids:
        mask = (img == class_id).astype(np.uint8)
        mask = binary_closing(mask, footprint=np.ones((3, 3)))
        skeleton = skeletonize(mask)
        endpoints=find_endpoints(skeleton)
        selected=pick_two_farthest_points(endpoints)
        skeletons[class_id] = skeleton
        all_skeletons[skeleton > 0] = class_id
        endpoints_dict[class_id]=selected
        
    endo_skeleton = skeletons[2]
    cervix_skeleton = skeletons[4]
    fundus_skeleton = skeletons[3]

    #cervix 스켈레톤 너무 짧은 경우
    if len((endpoints_dict[4]))==1:
        cervix_start=endpoints_dict[4][0]
    else:
        farthest_points=pick_two_farthest_points(np.array(endpoints_dict[4]))
        cervix_start, cervix_end=farthest_points[0],farthest_points[1]

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
    """
    선 하나 긋기 (큰 스켈레톤)
    """

    mask_fundus = (img == 3).astype(np.uint8)
    mask_cervix = (img == 4).astype(np.uint8)
    mask_endo = (img == 2).astype(np.uint8)
    mask_uterus = fill_mask(img).astype(np.uint8)
    endo_coords = sorted([tuple(coord) for coord in np.argwhere(endo_skeleton)], key=lambda x: x[0])
    cervix_coords = sorted([tuple(coord) for coord in np.argwhere(cervix_skeleton)], key=lambda x: x[0])
    fundus_coords = sorted([tuple(coord) for coord in np.argwhere(fundus_skeleton)], key=lambda x: x[0])
    endo_center=np.mean(endo_coords,axis=0)
    boundary_points = []
    connect_points = []
    
    if len(endpoints_dict[4]) == 1:
        cervix_point = endpoints_dict[4][0]
        if calc_distance(cervix_point,endpoints_dict[2][0])<=calc_distance(cervix_point,endpoints_dict[2][1]):
            endo_point = endpoints_dict[2][0]  # endo 스켈레톤 시작
        else:
            endo_point = endpoints_dict[2][1]
        base_point, direction, uterus_center = get_direction_for_short_skeleton(cervix_point, endo_point, mask_uterus)
        boundary_points.append(cervix_point)
        boundary_points.append(endo_point)
    else:
        # 기존 방식: 스켈레톤이 2픽셀 이상일 때
        base_point, direction, uterus_center = get_direction_with_slope_and_uterus(
            cervix_start,cervix_end, mask_uterus
        )
        boundary_points.append(cervix_start)
        boundary_points.append(cervix_end)
    if base_point is not None:
        cervix_boundary = extend_to_boundary(mask_uterus, base_point, direction)
        boundary_points.append(cervix_boundary)
    # cervix_coords의 끝점에서 endo_center와 반대 방향으로 연장
    sorted_cervix = sorted(cervix_coords, key=lambda x: x[0])


    boundary_points = [convert_to_int_tuple(point) for point in boundary_points]
    for point in boundary_points:
        cv2.circle(out_img, (point[1], point[0]), 2, (255, 0, 0), -1)
    connect_points.append(cervix_boundary)
    endpoints_dict = reorder_endpoints_by_reference(endpoints_dict, cervix_boundary)
    class_ids=[4,2,3]
    for num in class_ids:
        for point in endpoints_dict[num]:
            connect_points.append((int(point[0]),int(point[1])))
    skeleton_image=(all_skeletons>0).astype(np.uint8)
            

    
    sorted_points = sort_points_greedy(connect_points)
    print(f"connect points: {sorted_points} length of connect points: {len(sorted_points)}")
    out_img,all_coords=custom_connect(out_img,connect_points)
    img=np.array(img_pil.convert('RGB'))

    
    results=[
        f"{image_name} results",
        f"cervix start:{cervix_start} cervix end:{cervix_end} cervix point:{cervix_point} endo point:{endo_point}",
        f"length of connect points: {len(connect_points)}"
    ]
    for class_id, endpoints in endpoints_dict.items():
        for point in endpoints:
            cv2.circle(out_img, (int(point[1]), int(point[0])), 1, (0, 0, 255), -1)

    # #최대 내접원 찾기
    # (endo_center_x, endo_center_y), endo_diameter = find_largest_incircle(mask_endo,image_name,save_dir="endo_circle")
    # (uterus_center_x, uterus_center_y), uterus_diameter = find_largest_incircle(mask_uterus,image_name,save_dir="uterus_circle")
    # result = find_largest_inscribed_circle(mask_endo, endo_skeleton)
    # uterus_result = find_largest_inscribed_circle(mask_uterus, combined_skeleton)
    # #엔도 스켈레톤 이진화> 컨투어 추출> 길이계산
    # endo_skeleton=(endo_skeleton>0).astype(np.uint8)
    # contours,_=cv2.findContours(endo_skeleton,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # if contours:
    #     main_contour=max(contours,key=lambda x:cv2.arcLength(x, False))
    #     length_endo=cv2.arcLength(main_contour,closed=False)
    # else:
    #     length_endo=0
    
    
    
    # 스켈레톤 시각화 및 저장
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    # ax = axes.ravel()

    # ax[0].imshow(img_pil)
    # ax[0].axis('off')
    # ax[0].set_title('original', fontsize=20)

    # ax[1].imshow(out_img)
    # ax[1].axis('off')
    # ax[1].set_title('Combined skeleton', fontsize=20)

    # ax[1].legend(loc='upper right', fontsize=12)
    # plt.tight_layout()

    # base_name = os.path.splitext(image_name)[0]
    # save_path = os.path.join(output_dir, f"{base_name}_allSk.png")
    # fig.savefig(save_path)
    # plt.close(fig)

    # print(f"Saved skeleton image for {image_name} at {save_path}")
    
    # with open('results.txt','a') as f:
    #     for line in results:
    #         f.write(line+'\n')
    
    
    
    #4가지 두께 재는 방법 비교 저장    
    steps=[2,5,10]
    img_list=[]
    thickness_list=[]
    for step in steps:
        img_copy=img.copy()
        thickness=measure_uterus_thickness(mask_endo,all_coords,img_copy,step=step,probe_half_length=200)
        img_list.append(img_copy)
        thickness_list.append(thickness)

   
    # 시각화 및 저장
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img)
    ax[0].set_title('Original')
    ax[0].axis('off')

    # 나머지 칸: step별 결과 이미지
    for i, (img, step, thickness) in enumerate(zip(img_list, steps, thickness_list), start=1):
        ax[i].imshow(img)
        ax[i].set_title(f'Step={step} (thickness={thickness})')
        ax[i].axis('off')

    plt.tight_layout()
    base_name = os.path.splitext(image_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}_thickness.png")
    fig.savefig(save_path)
    plt.close(fig)



    print(f"Saved skeleton image for {image_name} at {save_path}")
    
    with open('results.txt','a') as f:
        for each in results:
            f.write(each+'\n')
    





