from tensorflow.keras import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Lambda, Convolution2D, MaxPooling2D, Dense, Dropout,Flatten

def AlexNet(num_classses):
    ##构建网络
    model = Sequential()
    model.add(ZeroPadding2D((2, 2), input_shape=(160, 160, 3)))
    # model.add(Lambda(lambda x: x / 255.0))  # 归一化
    model.add(Convolution2D(64, (11, 11), strides=(4, 4), activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(192, (5, 5), activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(384, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classses, activation='softmax'))

    return model