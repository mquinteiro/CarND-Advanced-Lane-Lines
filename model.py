import csv
print("Loading cv2...")
import cv2
import numpy as np
print("Loading tensorflow...")
import numpy as np
import tensorflow as tf
print("Loading numpy...")
from glob import glob

print("Loading keras...")
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPool2D, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import random


DATA_PATH = "/home/mquinteiro/proyectos/CarND-Advanced-Lane-Lines/train/"

#the image loading and augment augmentation
def loadImages(path):
    print("Loading images...")

    lines = []
    labels = []

    trues = glob(path+"line_*.png")
    for imgFile in trues:
        image = cv2.imread(imgFile)
        if image is None or image.shape!=(40,50,3):
            continue
        lines.append(image)
        lines.append(np.fliplr(image))
        labels.append([1,0])
        labels.append([1, 0])
    trues = glob(path + "other_*.png")
    for imgFile in trues:
        image = cv2.imread(imgFile)
        if image is None or image.shape!=(40,50,3):
            continue
        lines.append(np.fliplr(image))
        #print image.shape, imgFile
        lines.append(image)
        labels.append([0, 1])
        labels.append([0, 1])
    return np.array(lines), np.array(labels)

#this function is a wrapper to select witch architecture we will use.
def defineDriver(model='MQ'):
    if model=='MQ' :
        modelF = mqModel
    elif model=='NVidia':
        modelF = nVidiaModel
    else:
        print("Model "+ model + " not found")
        exit(-1)

    return modelF()


def modelDef():
    print("Creating NVidea model...")
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(40,50, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam")
    #model.compile(loss='mse', optimizer=Adam()
    return model

def trainDriver(model, X_train,y_train, epochs):
    print("Training the model...")
    model.fit(X_train,y_train,validation_split=0.20, shuffle=True,epochs=epochs)


import matplotlib.pyplot as plt
def main():
    #parameters for histograms
    plt.ion()
    bin_size = 0.1;
    min_edge = -0.95
    max_edge = 0.95
    N = 21


    # create de model
    import os.path as path
    from keras.models import load_model
    if path.exists("model.h5"):
        model = load_model("model.h5")
    else:
        model = modelDef()
        X_train, y_train = loadImages(DATA_PATH)
        trainDriver(model, X_train,y_train,8)
        model.save('model.h5')

    for i in range (1,12):
        #example = cv2.imread("test_images/test"+str(i)+".jpg")
        filename = "test_images/test" + str(i) + ".jpg"
        #filename = "shot00{:02.0f}.png".format(i)
        example = cv2.imread(filename)
        windowed = []
        for r in range(360,example.shape[0]-41,20):
            for c in range(0,example.shape[1]-50,25):
                windowed.append(example[r:r+40,c:c+50])

        result = model.predict(np.array(windowed), batch_size=1)
        idx=0
        big = result[:,0]<result[:,1]
        for r in range(360,example.shape[0]-41,20):
            for c in range(0,example.shape[1]-50,25):
                if big[idx]:
                    cv2.rectangle(example, (c, r), (c + 50, r + 40), (0, 0,255))
                else:
                    cv2.rectangle(example, (c, r), (c + 50, r + 40), (0, 255, 0))
                idx+=1
        cv2.imshow("test",example)
        cv2.waitKey(0)
    print(result)
    # old comvinations of different datasets.

    #X_train, y_train = loadImages(DATA_PATH + "C2TN1/", 2 / 25.0)
    #X_train = np.concatenate([X_train, X_train2], axis=0)
    #y_train = np.concatenate([y_train, y_train2], axis=0)
    #totalLabels = np.concatenate([totalLabels, y_train], axis=0)
    #plt.hist(y_train, bin_list)
    '''plt.draw()
    trainDriver(model, X_train, y_train, 8)
    model.save('model.h5')
    exit(0)
    X_train, y_train = loadImages(DATA_PATH + "PC1/", 5 / 25.0)
    totalLabels = np.concatenate([totalLabels,y_train],axis=0)
    plt.hist(y_train, bin_list)
    trainDriver(model, X_train, y_train, 2)
    model.save('model.h5')
    X_train, y_train = loadImages(DATA_PATH+"2N/", 5 / 25.0)
    totalLabels = np.concatenate([totalLabels, y_train],axis=0)
    trainDriver(model, X_train, y_train, 2)
    X_train, y_train = loadImages(DATA_PATH + "linux_sim/linux_sim/", 5 / 25.0)
    totalLabels = np.concatenate([totalLabels, y_train],axis=0)
    plt.hist(y_train, bin_list)
    plt.draw()
    trainDriver(model, X_train, y_train, 2)
    model.save('model.h5')
    X_train, y_train = loadImages(DATA_PATH + "CURVES/", 5 / 25.0)
    totalLabels = np.concatenate([totalLabels, y_train],axis=0)
    plt.hist(y_train, bin_list)
    plt.draw()
    trainDriver(model, X_train, y_train, 5)
    model.save('model.h5')
    '''




if __name__ == '__main__':
    main()


