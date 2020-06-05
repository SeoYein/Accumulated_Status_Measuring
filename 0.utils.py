from __future__ import absolute_import, division, print_function

import cv2
from imutils import paths
import os
import shutil
from PIL import Image, ImageOps
import time
import random
#from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFilter
from PIL import ImageEnhance

# utils: 끌어다쓰는 함수 모음, 윈도우 기준
# help(함수명) 혹은 함수명.__doc__ 을 실행하여 함수 정보 확인 가능


# 디렉토리 존재 여부 확인 뒤 디렉토리 만들기
def dir_make(dir_name):
    """
    디렉토리 존재 여부 확인 뒤 디렉토리 만들기
    :param dir_name: 만들려고 하는 디렉토리명 (str)
    """
    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    except OSError :
        print('Creating directory...')


# 특정 디렉토리의 파일명 리스트로 가져오기
def file_nm(image_directory):
    """
    특정 디렉토리 입력 시 그 디렉토리 안의 파일명 리스트로 가져오기
    :param image_directory: 파일명을 가져올 디렉토리명 (str)
    :return: 파일들의 절대경로가 담긴 리스트
    """
    file_list = []
    for imagepath in paths.list_images(image_directory):
        file_list.append(imagepath)
    return file_list


# 이미지 이름 목록(리스트)을 받으면 지정한 디렉토리에 복사
# copying, not moving
def image_to_folder(name_list, dir_name) :
    """
    이미지 이름 목록(리스트)을 받으면 지정한 디렉토리에 복사 후 "디렉토리명 is created"를 프린트
    :param name_list: 이미지 파일 이름 리스트
    :param dir_name: 파일을 복사하려고 하는 디렉토리 (없는 경우엔 만듦)
    """
    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    except OSError :
        print('Creating directory...')

    for nm in name_list :
        name = nm.split(os.sep)[-1]
        change_dir = dir_name
        shutil.copy(nm, os.path.join(change_dir, name))
    print(dir_name, "is created")



# blur 이미지인지 thresholding 하기 위한 값
def get_blur(img_name):
    """
    1) blur 이미지인지 판단을 위해 파일명을 받아 이미지를 읽음
    2) 이미지를 그레이스케일로 변환
    3) 라플라시안 분산을 구해 반환
    :param img_name: 라플라시안 분산을 구하려는 이미지 파일명
    :return: 라플라시안 분산값
    """
    image = cv2.imread(img_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.Laplacian(gray, cv2.CV_64F).var()
    return result