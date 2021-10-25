#在测试集中判断多少预测中
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from tensorflow.python.keras.models import load_model
import utils
import cv2
from sklearn.metrics import accuracy_score
CNN_MODEL="./model/new.h5"
OP_DATASET="./op_dataset"

face_recognition_model = load_model(CNN_MODEL)
raw_images, raw_labels = utils.read_image_to_train(OP_DATASET)
raw_images = np.asarray(raw_images, dtype=np.int8)
raw_images = np.asarray(raw_images, dtype=np.float32)
raw_images /= 255.0
lables = face_recognition_model.predict_classes(raw_images)

right=0.0
for i in range(0,len(lables)):

    print(raw_labels[i])
    if lables[i]==int(raw_labels[i]):
        right+=1

print("right"+str(right/len(lables)))