import os
import cv2
import utils
PRE_DATASET = "./pre_dataset"
OP_DATASET = "./op_dataset"

if __name__ == '__main__':
    # 遍历pre_dataset
    for i in os.listdir(PRE_DATASET):  # i为姓名
        input_dir = PRE_DATASET + "/" + i  # ./pre_dataset/zhoujielun
        output_dir = OP_DATASET + "/" + i
        #如果out_dir不存在建立该文件夹
        folder = os.path.exists(output_dir)
        if not folder:
            os.mkdir(output_dir)
        _index = 0
        for j in os.listdir(input_dir):
            image_path = input_dir + "/" + j #./pre_dataset/zhoujielun/7.jpg
            image = cv2.imread(image_path)
            image,images,tmp=utils.face_detector_dnn(image)
            if images:
                for face in images:
                    #将人脸存回去
                    face=utils.resize_image(face)
                    _index+=1
                    cv2.imwrite(output_dir+"/%d.jpg"%_index,face)
                    print("存放"+output_dir+"/%d.jpg"%_index)
                    #调用识别以及裁剪函数

