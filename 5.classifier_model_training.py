from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

from sklearn.ensemble import RandomForesstClassifier
from tensorflow.keras.models import Model
import pickle

from utils import *

########################
# 고쳐야 할 값
## normalize 값
## 이름, weight
########################

## 디렉토리 설정
train_dir = "/home/pirl/posco/Dataset/training_120000_roi_posco_2_relabeled_11class"
# test dir = # 테스트 시 작성
np_save_dir = "./feature_np_save/nasnet_120000_roi_posco_2_relabeled_11class_50"
classifier_save_name = 'rf_roi_posco_2_120000_clear_relabeled_11class_50ckpt.pkl'

## 변수 설정
weight_path = "/home/pirl/posco/code/weight/120000_clear_relabeled_11class/50.ckpt"
IMAGE_SIZE = (331,331)
class_num = 11

# normalize
mean = [0.29852340278693823, 0.3134894330980739, 0.3181296518181414] # 120000 roi_posco_2 relabeled 10 class
mean[0] = mean[0] * 255
mean[1] = mean[1] * 255
mean[2] = mean[2] * 255
std = [0.20286728485059027, 0.20229726063160275, 0.19632965079569015] # 120000 roi_posco_2 relabeled 10 class
std[0] = std[0] * 255
std[1] = std[1] * 255
std[2] = std[2] * 255

def normalize(image):
    """
    z-normalization 진행
    mean, std값은 함수 외부에서 지정
    :param image: nomalize할 이미지
    :return: normalize된 이미지의 array 값
    """
    image = np.array(image)
    image[0] = [x - mean[0] for x in image[0]]
    image[1] = [x - mean[1] for x in image[1]]
    image[2] = [x - mean[2] for x in image[2]]
    image[0] = [x / std[0] for x in image[0]]
    image[1] = [x / std[1] for x in image[1]]
    image[2] = [x / std[2] for x in image[2]]
    return image

def preprocess(nm,IMAGE_SIZE):
    """
    적치량 측정 모델에 넣기 위한 전처리 과정
    RGB로 변환하고 resize 후 normalization 및 rescale 진행 이후 모델에 투입하기 위한 형태로 reshape
    :param nm: 이미지 파일명
    :param IMAGE_SIZE: 모델에 넣기 위해 resize하기 위한 이미지 크기 리스트 혹은 튜플 (nasnet의 경우 (331,331))
    :return: 모델 투입을 위해 가공된 이미지
    """
    img = cv2.imread(nm)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_resize = cv2.resize(img2, dsize=IMAGE_SIZE, interpolation = cv2.INTER_NEAREST) # keras랑 interpolation 방식 맞춤
    im_norm = normalize(im_resize)
    im_rescale = im_norm / 255
    im_reshape = im_rescale.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[0], 3)
    return im_reshape


## 학습된 feature 추출용 모델(vgg) 불러오기
K.clear_session()
K.set_learning_phase(False)

base_model = tf.keras.applications.NASNetLarge(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                                   include_top=False,
                                                   weights='imagenet')


feature_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(class_num, activation='softmax')])


feature_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# initialize : for using tf.hub initialize is needed to be manually
sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

# weight load
feature_model.load_weights(weight_path)
feature_model.summary()
layer_name = 'global_average_pooling2d'

# feature extractor 지정
feature_extractor = Model(inputs=feature_model.input,
                          outputs=feature_model.get_layer(layer_name).output)

## 데이터 가져오기
train_files = sorted(file_nm(train_dir))

# label
labels = sorted(os.listdir(train_dir))
label_number = len(os.listdir(train_dir))
train_label = np.zeros(shape=len(train_files))

# feature extraction
train_features = np.zeros(shape = (len(train_files),4032))


#### train file 피처 추출
print('train feature extract start ...')
count = 0
start = time.time()
for i, files in enumerate(train_files):
    # image feature
    img_np = preprocess(files, IMAGE_SIZE)
    img_feature = feature_extractor.predict(img_np)
    train_features[i] = img_feature

    # label
    directory_name = os.path.dirname(files).split(os.sep)[-1]
    label_idx = labels.index(directory_name)
    train_label[i] = label_idx
    count += 1
    if count%500 == 0:
        print('done file number: ', count, 'file:', files)

print('train feature extract done')
print('spend time: ', (time.time() - start) // 60)


### 테스트 피처 추출
# print('train feature extract start ...')
# test_files = sorted(file_nm(test_dir))
# labels = sorted(os.listdir(test_dir))
# test_label = np.zeros(shape=len(test_files))
#
# test_features = np.zeros(shape = (len(test_files),4032))
#
# start = time.time()
# count = 0
# for i, files in enumerate(test_files):
#     # image feature
#     img_np = preprocess(files, IMAGE_SIZE)
#     img_feature = feature_extractor.predict(img_np)
#     test_features[i] = img_feature
#
#     # label
#     directory_name = os.path.dirname(files).split(os.sep)[-1]
#     label_idx = labels.index(directory_name)
#     test_label[i] = label_idx
#     count += 1
#     if count % 100 == 0:
#         print('done file number: ', count)
#
# print('test feature extract done')
# print('spend time: ', (time.time() - start) // 60)
#
# # 추출한 테스트 피처 저장
# np.save(os.path.join(np_save_dir,'test_features'),test_features)
# np.save(os.path.join(np_save_dir,'tset_label'),test_label)
# print('test feature save done in', np_save_dir)
# print('         ')


#### 저장한 피처값 로딩하고 싶을 때
# train_features = np.load(os.path.join(np_save_dir, train_features.npy))
# train_label = np.load(os.path.join(np_save_dir, train_label.npy))
# test_features = np.load(os.path.join(np_save_dir, test_features.npy))
# test_label = np.load(os.path.join(np_save_dir, test_label.npy))


### rf classifier
print('classifier training start...')
rf = RandomForestClassifier(criterion='gini', max_depth=None, n_estimators=100, random_state=42, max_features=5)
rf.fit(train_features, train_label)
print('classifier training done')

# 테스트 결과 보기
# result = rf.score(test_features, test_label)
# print('result: ', result)
# print('classifier save start...')


### classifier 저장
with open(classifier_save_name, 'wb') as f:
    pickle.dump(rf, f)
print('classifier save done: ', classifier_save_name )


## 추출한 피처값 향후 rf 모델 테스트를 위해 저장
dir_make(np_save_dir)
np.save(os.path.join(np_save_dir,'train_features'),train_features)
np.save(os.path.join(np_save_dir,'train_label'),train_label)
print('train feature save done in', np_save_dir)
print("         ")



