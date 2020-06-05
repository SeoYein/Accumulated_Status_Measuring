from __future__ import absolute_import, division, print_function

from utils import *


########################
# 데이터 디렉토리
########################
root_dir = "D:\\(0806)new_data\\all"
train_dir = "D:\\(0806)new_data\\all_dataset\\Train"
valid_dir = "D:\\(0806)new_data\\all_dataset\\Valid"
test_dir = "D:\\(0806)new_data\\all_dataset\\Test"

# train/val/test data의 비율
train_portion = 0.75
valid_portion = 0.05
test_portion = 0.2

# 디렉토리 존재 여부 확인하고 없으면 만들기
dir_make(train_dir)
dir_make(valid_dir)
dir_make(test_dir)


# 클래스별로 폴더를 가져와서 shuffle후 슬라이싱해서 나눈 뒤 저장
shuffler = random.Random(2019)


for dir in os.listdir(root_dir): # dir : 하나의 클래스
    files = file_nm(root_dir + os.sep + dir)
    shuffler.shuffle(files)
    train_files = files[:int(len(files)*train_portion)]
    valid_files = files[int(len(files)*train_portion):int(len(files)*(train_portion+valid_portion))]
    test_files = files[int(len(files)*(train_portion+valid_portion)):]
    image_to_folder(train_files, train_dir + os.sep + dir)
    image_to_folder(valid_files, valid_dir + os.sep + dir)
    image_to_folder(test_files, test_dir + os.sep + dir)


