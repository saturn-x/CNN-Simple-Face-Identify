#工具包

#实现人脸检测以及人脸裁剪
import os

import numpy as np
from copy import deepcopy
import cv2
modelPath = "./model/deploy.prototxt.txt"
caffePath = "./model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
IMAGESIZE=160
confidence = 0.3 # 置信度参数，高于此数认为是人脸
def face_detector_dnn(image):

    image_1 = deepcopy(image)
    net = cv2.dnn.readNetFromCaffe(modelPath, caffePath)
    # 输入图片并重置大小符合模型的输入要求
    (h, w) = image.shape[:2]  #获取图像的高和宽，用于画图
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()  # 预测结果
    face_img = []
    rect_position=[]
    # 可视化：在原图加上标签和框
    for i in range(0, detections.shape[2]):
        # 获得置信度
        res_confidence = detections[0, 0, i, 2]
        # 过滤掉低置信度的像素
        if res_confidence > confidence :
            # 获得框的位置
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            rect_position.append(np.array([startX, startY, endX, endY]))
            # 在图片上写上标签
            # text = "{:.2f}%".format(res_confidence * 100)
            # 如果检测脸部在左上角，则把标签放在图片内，否则放在图片上面
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(image, (startX, startY), (endX, endY),
            #     (0, 255, 0), 2)
            # cv2.putText(image, text, (startX, y),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            if len(deepcopy(image_1)[startY:endY, startX:endX]) != 0:
                face_img.append(deepcopy(image_1)[startY:endY, startX:endX])

    if len(face_img) == 0:
        return None, None,None
    else:
        # return image, face_img
        return image,face_img,rect_position

def resize_image(image, height=IMAGESIZE, width=IMAGESIZE):

    top, bottom, left, right = (0, 0, 0, 0)

    # 获取图像尺寸
    h, w, _ = image.shape
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)
    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    # RGB颜色
    BLACK = [0, 0, 0]
    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))


#该函数主要是训练时候读取照片到内存中
def read_image_to_train(Path):
    data_x, data_y = [], []
    num = 0
    for root, dirs, files in os.walk(Path):
        if len(files) > 5:
            for i in files:
                ImgPath = root + "/" + i
                im = cv2.imread(ImgPath)
                data_x.append(np.asarray(im, dtype=np.int8))
                data_y.append(str(num))
            num += 1
    print("读取照片完成")
    return data_x, data_y




# if __name__ == '__main__':
#     #测试一张照片
#     image_path="./pre_dataset/denglun/1.jpg"
#     image=cv2.imread(image_path)
#     cv2.imshow("denglun",image)
#     image,images=face_detector_dnn(image)
#     for i in images:
#         cv2.imshow("dele ",i)
#         i=resize_image(i)
#         cv2.imshow("dele ",i)
#     cv2.waitKey(0)