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

    cervix_start, cervix_end=tuple(np.argwhere(endpoints_dict[3]))[0] , tuple(np.argwhere(endpoints_dict[3]))[1]
    fundus_start, fundus_end=tuple(np.argwhere(endpoints_dict[4]))[0] , tuple(np.argwhere(endpoints_dict[4]))[1]


        
    
    img_pil = Image.fromarray(img.astype(np.uint8), mode="P")
    img_pil.putpalette(palette)
    img_pil=img_pil.convert('RGB')
    

    
    combined_mask=np.isin(img,[2,3,4]).astype(np.uint8)
    combined_mask=binary_closing(combined_mask,footprint=np.ones((3,3)))
    skeleton=skeletonize(combined_mask)
    out_img = Image.fromarray(all_skeletons, mode="P")
    out_img.putpalette(palette)
    out_img=np.array(out_img.convert('RGB'))
    
    #선 두개 긋기
    mask_fundus = (img == 4).astype(np.uint8)
    mask_cervix = (img == 3).astype(np.uint8)

    transition_fundus = extend_line_find_transition(mask_fundus, fundus_start, fundus_end, direction='end')
    transition_cervix = extend_line_find_transition(mask_cervix,cervix_start, cervix_end, direction='start')
    print(f"transition_fundus:{transition_fundus}, transition_cervix:{transition_cervix}")
    
    cv2.line(out_img,yx2xy(fundus_start),yx2xy(transition_fundus),(255,0,0),1)
    cv2.line(out_img,yx2xy(transition_cervix),yx2xy(cervix_end),(255,0,0),1)
    
    #길이 재기
    length_fundus=calc_distance(fundus_start,transition_fundus)
    length_cervix=calc_distance(transition_cervix,cervix_end)
    
    #엔도 스켈레톤 이진화> 컨투어 추출> 길이계산
    endo_skeleton=(endo_skeleton>0).astype(np.uint8)
    contours,_=cv2.findContours(endo_skeleton,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if contours:
        main_contour=max(contours,key=lambda x:cv2.arcLength(x, False))
        length_endo=cv2.arcLength(main_contour,closed=False)
    else:
        length_endo=0
    print(f"fundus 길이 : {length_fundus}, cervix 길이: {length_cervix}, endo 길이: {length_endo}")
    result_str = (
    f"transition_fundus:{transition_fundus}, transition_cervix:{transition_cervix}\n"
    f"fundus 길이 : {length_fundus}, cervix 길이: {length_cervix}, endo 길이: {length_endo}\n"
    f"\n"
    )

    
    
    
    
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
    
    with open("result.txt", "a", encoding="utf-8") as f:
        f.write(result_str)





