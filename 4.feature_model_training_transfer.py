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

from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model

from utils import *


# NAS NET
IMAGE_SIZE = [331, 331]

## VGG NET, ResNET
# IMAGE_SIZE = [224,224]

## Inception resnet v2
#IMAGE_SIZE = [299,299]

K.clear_session()

########### 기존걸 가져와서 재학습 시 #######
Transfer = True # 재학습 여부
weight_path = ".\\weight\\30.ckpt" # 재학습 시키려는 weight가 있는 경로 및 weight 이름
##########################################

BATCH_SIZE = 8
# BATCH_SIZE = 32
EPOCHS = 1
patience = 20


tf.set_random_seed(3)

train_dir = "D:\\0718_dataset\\Train"
valid_dir = "D:\\0718_dataset\\Valid"
model_path = "D:\\(0806)new_data"
model_name = "test_1.h5"
checkpoint_path = 'training_1/{epoch:02d}.ckpt'

dir_make(model_path)

print(train_dir)
print(valid_dir)
print('batch:',BATCH_SIZE)
print('epoch:',EPOCHS)
print('lr:', '0.001')
print(model_path)
print(model_name)
print("loss: categorical")


mean = [0.31745544292830674, 0.3320354514434907, 0.3312002839781163]
mean[0] = mean[0] * 255
mean[1] = mean[1] * 255
mean[2] = mean[2] * 255
std = [0.19934886738836136, 0.19493450113345362, 0.17991260461583236]
std[0] = std[0] * 255
std[1] = std[1] * 255
std[2] = std[2] * 255

def normalize(image):
    image = np.array(image)
    image[0] = [x - mean[0] for x in image[0]]
    image[1] = [x - mean[1] for x in image[1]]
    image[2] = [x - mean[2] for x in image[2]]
    image[0] = [x / std[0] for x in image[0]]
    image[1] = [x / std[1] for x in image[1]]
    image[2] = [x / std[2] for x in image[2]]
    return image

## Data
# 이미지 데이터 generator 만들기 (for Data augmentation)

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                  preprocessing_function=normalize)
                                                                  #rotation_range= 0.05,
                                                                  #width_shift_range=0.05,
                                                                  #height_shift_range= 0.025)

# train set 가져오는 함수, batch_size = 32
train_set = train_generator.flow_from_directory(train_dir,
                                                target_size = IMAGE_SIZE,
                                                batch_size= BATCH_SIZE,
                                                class_mode='categorical')

valid_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                  preprocessing_function=normalize)

valid_set = valid_generator.flow_from_directory(valid_dir,
                                                target_size=IMAGE_SIZE,
                                                batch_size= BATCH_SIZE,
                                                class_mode='categorical')


## 학습데이터 정보 확인
# 디렉토리에서 class 이름 정보 가져오기 (subdirectory 명)
class_name = []
for dir in os.listdir(train_dir):
    if os.path.isdir(os.path.join(train_dir, dir)):
        class_name.append(dir)

# train set 정보 출력 : 어떤 클래스에 몇 장씩 있는지
for nm in class_name:
    print("class label name: ", nm, "  count: ", len(file_nm(train_dir + os.sep + nm)))
print(train_set.class_indices)

## input shape 확인
for image_batch, label_batch in train_set :
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break


## Model
# 모델 구성
K.set_learning_phase(True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# NASNet : https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-large-no-top.h5
base_model = tf.keras.applications.NASNetLarge(input_shape= (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                         include_top=False,
                                         weights='C:\\Users\\seoyein\\.keras\\models')

# base_model = tf.keras.applications.VGG16(input_shape= (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
#                                          include_top=False,
#                                          weights='imagenet')

# base_model = tf.keras.applications.InceptionResNetV2(input_shape= (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
#                                          include_top=False,
#                                          weights='imagenet')

base_model.trainable = True
print("trainable:", "True")


# feature extraction 용
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(train_set.num_classes, activation='softmax')])

model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])


if Transfer==True:
    model.load_weights(weight_path)
    print("weight loading done")


model.summary()

print("trainable_variables:", len(model.trainable_variables))

# 학습 progress 보기
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])

# gpu 활용 가능 여부 확인
print("GPU using: ", tf.test.is_gpu_available())

steps_per_epoch = train_set.samples//train_set.batch_size

batch_stats = CollectBatchStats()

## 체크포인트 저장
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# early stopping
early_stopping= EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='auto')


print("training...")
## 모델 학습
model.fit_generator(train_set,
                steps_per_epoch=steps_per_epoch,
                epochs=EPOCHS,
                validation_data = valid_set,
                validation_steps = valid_set.samples//valid_set.batch_size,
                callbacks = [cp_callback, early_stopping],
                shuffle=True)



print("Training done")

# 모델 저장
# tf.keras.models.save_model(
# model,
# os.path.join(model_path, model_name),
# overwrite=True,
# include_optimizer=False
# )

model.save(model_name)
print("Feature Model Saved:", os.path.join(model_path, model_name))




