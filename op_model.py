#该文件主要操作AlexNet
import numpy as np
from keras.optimizer_v2.gradient_descent import SGD
from keras.utils import np_utils
from sklearn.model_selection import  train_test_split
import  AlexNet
import utils
OP_DATASET="./op_dataset"
#存放训练完模型的路径
MODEL_PATH = './model/new.h5'
#模型的有多少类，也就是图片的标签个数
CLASSES=11
IMAGE_SIZE = 160

def train():
    print("开始读照片···")
    raw_images, raw_labels = utils.read_image_to_train(OP_DATASET)
    raw_images, raw_labels = np.asarray(raw_images, dtype=np.float32), np.asarray(raw_labels, dtype=np.int32)
    ont_hot_labels = np_utils.to_categorical(raw_labels)
    train_input, valid_input, train_output, valid_output = train_test_split(raw_images,
                                                                            ont_hot_labels,
                                                                            test_size=0.3)
    train_input /= 255.0
    valid_input /= 255.0
    model = AlexNet.AlexNet(CLASSES)
    learning_rate = 0.01
    decay = 1e-6
    momentum = 0.8
    nesterov = True
    sgd_optimizer = SGD(lr=learning_rate, decay=decay,
                        momentum=momentum, nesterov=nesterov)
    model.compile(loss = 'categorical_crossentropy',
                               optimizer = sgd_optimizer,
                               metrics = ['accuracy'])
    batch_size = 20  # 每批训练数据量的大小
    epochs =20
    print("开始训练···")
    model.fit(train_input, train_output,
                               epochs=epochs,
                               batch_size=batch_size,
                               shuffle=True,
                               validation_data=(valid_input, valid_output))
    print(model.evaluate(valid_input, valid_output, verbose=0))

    model.save(MODEL_PATH)

train()










