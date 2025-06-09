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
from skimage.draw import line


def find_major_axis_pca(mask):
    """
    axis_vector: 장축 방향 단위 벡터
    length: 장축 길이
    endpoints: 장축 양끝점
    """
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)==0:
        return None, None, None, None
    points=contours[0].reshape(-1,2)
    
    mean=np.mean(points,axis=0)
    centered=points-mean
    cov=np.cov(centered,rowvar=False)
    eigvals,eigvecs=np.linalg.eigh(cov)
    order=np.argsort(eigvals)[::-1]
    pc1=eigvecs[:,order[0]]
    pc2=eigvecs[:,order[1]]
    
    projections=centered @ pc1
    min_proj, max_proj = projections.min(), projections.max()
    length=max_proj-min_proj
    
    endpoint1=mean+pc1*max_proj
    endpoint2=mean+pc1*min_proj
    
    return tuple(mean),pc1,pc2,length,(tuple(endpoint1),tuple(endpoint2))


#pca 로 구한 장축 그리는 함수
def draw_pca_result(mask, image_name="default",save_dir="pca_results"):
    os.makedirs(save_dir, exist_ok=True)
    filled_mask=binary_fill_holes(mask>0).astype(np.uint8)*255
    center,pc1, pc2, length,(endpoint1,endpoint2)=find_major_axis_pca(filled_mask)
    if center is None:
        print("윤곽선이 없어서 그릴수 없습니다")
        return    
    h,w=filled_mask.shape
    result_img=np.zeros((h,w,3),dtype=np.uint8)
    result_img[filled_mask==255]=[255,0,0]
    
    pc1=tuple(np.round(endpoint1).astype(int))
    pc2=tuple(np.round(endpoint2).astype(int))
    
    cv2.line(result_img,pc1,pc2,(0,0,255),2)
    save_path=os.path.join(save_dir,f"{image_name}_major_axis.png")
    cv2.imwrite(save_path,result_img)
    print(f"pca결과 이미지 저장 완료:{save_path}")
    
    return save_path

