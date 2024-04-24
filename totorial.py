import torch
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import timm
import matplotlib.pyplot as plt
# Label Encoding
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import itertools
import tensorflow as tf
# confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # read train
    train = pd.read_csv("./train.csv")
    print(train.shape)
    head = train.head(5)
    print(head)

    # put labels into y_train variable
    Y_train = train["label"]
    # Drop 'label' column
    X_train = train.drop(labels=["label"], axis=1)

    # plot some samples
    img = X_train.iloc[0].to_numpy()
    img = img.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.title(train.iloc[0, 0])
    plt.axis("off")

    X_train = X_train / 255.0
    # Reshape
    X_train = X_train.values.reshape(-1, 28, 28, 1)
    # Label Encoding

    X_train = torch.tensor(X_train)

    Y_train = to_categorical(Y_train, num_classes=10)

    print("x_train shape: ", X_train.shape)
    print("Y_train shape: ", Y_train.shape)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)
    print("x_train shape", X_train.shape)
    print("x_test shape", X_val.shape)
    print("y_train shape", Y_train.shape)
    print("y_test shape", Y_val.shape)

    model = Sequential()
    #
    model.add(Conv2D(filters=8, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    # fully connected
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    # Define the optimizer
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    epochs = 10  # for better result increase the epochs
    batch_size = 250

    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range=0.1,  # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    # Fit the model
    # 데이터 제너레이터를 사용한 모델 학습
    history = model.fit(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, Y_val),
        steps_per_epoch=X_train.shape[0] // batch_size
    )

    # Predict the values from the validation dataset
    Y_pred = model.predict(X_val)
    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_val, axis=1)
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    # plot the confusion matrix
    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
