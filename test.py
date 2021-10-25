import os

import numpy as np
from keras import backend as K
from keras.utils import np_utils
from tensorflow.python.keras.models import load_model
import utils
import cv2
from sklearn.metrics import accuracy_score
CNN_MODEL="./model/new.h5"
OP_DATASET="./op_dataset"
color = (0, 255, 0)
dict={}#标签对应的名字
def mark_face(frame,name,rect_position):
    cv2.rectangle(frame, (rect_position[0],rect_position[1]), (rect_position[2],rect_position[3]), color, thickness=2)
    cv2.putText(frame, name,
                (rect_position[0] + 30, rect_position[1] + 30),  # 坐标
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                1,  # 字号
                (255, 0, 255),  # 颜色
                2)
def video():

    # ip摄像头
    # video = "http://admin:110124@192.168.5.105:8081/"  # 此处@后的ipv4 地址需要修改为自己的地址
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("camera", 1)
    face_recognition_model = load_model(CNN_MODEL)
    tmp=0
    while cap.isOpened():
        # print("摄像头运行中···")
        ok, frame = cap.read()
        if not ok:
            break
        #通过frame识别出人脸
        face_frame,faces,rect_position=utils.face_detector_dnn(frame)
        if faces:
            print("识别到%d张人脸"%len(faces))
            for i in range(0,len(faces)):
                face=faces[i]
                face=utils.resize_image(face)#裁剪到160*160
                face = np.asarray(face, dtype=np.int8)
                face = np.asarray(face, dtype=np.float32)
                face /= 255.0
                face = np.reshape(face, (1,) + face.shape)
                lables=face_recognition_model.predict_classes(face)
                print("labels"+str(lables))
                mark_face(frame, dict[lables[0]], rect_position[i]) #写上标签
                print(face.shape)
                print("result_acc", np.around(face_recognition_model.predict(face), 2))
                # print("result", np.around(face_recognition_model.predict_proba(face), 2))
            cv2.imshow("shuaife",frame)
            cv2.waitKey(10)
        else:
            print("未识别到人脸")



tmp=0
for i in os.listdir("./op_dataset"):
    dict[tmp]=i
    tmp+=1
print("字典"+str(dict))
video()
# 释放摄像头并销毁所有窗口

cv2.destroyAllWindows()
