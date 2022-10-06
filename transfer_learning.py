##
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
import cv2

##
# load_dataset function to load the data and resize the images to 50x50
def load_dataset(directory):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory + '/' + label):
            filepath = directory + '/' + label + "/" + file
            img = cv2.resize(cv2.imread(filepath), (50, 50))
            images.append(img)
            labels.append(idx)
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels


##
# display_images function to show examples
def display_images(x_data, y_data, title, display_label=True):
    x, y = x_data, y_data
    fig, axes = plt.subplots(5, 8, figsize=(18, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.suptitle(title, fontsize=18)
    for i, ax in enumerate(axes.flat):
        ax.imshow(cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB))
        if display_label:
            ax.set_xlabel(uniq_labels[y[i]])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


##
# loading_dataset into X_pred and Y_pred
data_dir = r'C:\Users\USER\PycharmProjects\Sign Language Recognition(Final)\dataset'
uniq_labels = sorted(os.listdir(data_dir))
X_pred, Y_pred = load_dataset(data_dir)
print(X_pred.shape, Y_pred.shape)

##
# splitting dataset into 80% train, 10% validation and 10% test data
X_train, X_test, Y_train, Y_test = train_test_split(X_pred, Y_pred, test_size=0.8)
X_test, X_eval, Y_test, Y_eval = train_test_split(X_test, Y_test, test_size=0.5)

##
# print shapes and show examples for each set
print("Train images shape", X_train.shape, Y_train.shape)
print("Test images shape", X_test.shape, Y_test.shape)
print("Evaluate image shape", X_eval.shape, Y_eval.shape)
print("Printing the labels", uniq_labels, len(uniq_labels))
display_images(X_train, Y_train, 'Samples from Train Set')
display_images(X_test, Y_test, 'Samples from Test Set')
display_images(X_eval, Y_eval, 'Samples from Validation Set')

##
# converting Y_test and Y_train to One hot vectors using to_categorical
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_eval = to_categorical(Y_eval)

##
# Normalizing the data
X_train = X_train / 255
X_test = X_test / 255
X_eval = X_eval / 255

##
# building the transfer model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(50, 50, 3))
base_model.summary()
# len(base_model.layers)

##
i = 0
for layers in base_model.layers:
    layers.trainable = False
    i = i + 1
    if i > 19:
        break

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(24, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)
model.summary()
##
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="categorical_crossentropy",
              metrics=["accuracy"])
##
history = model.fit(X_train, Y_train, epochs=15, verbose=1, validation_data=(X_eval, Y_eval))
##
model.evaluate(X_test, Y_test)
##
# save the model
model.save('transfer_model2.h5')

##
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

##
# plotting training and validation loss vs. epochs
epochs = range(len(train_acc))
plt.plot(epochs, train_loss, label="training loss")
plt.plot(epochs, val_loss, label="validation  loss")
plt.legend()
plt.show()
##
# plotting training and validation accuracy vs. epochs
plt.plot(epochs, train_acc, label="training accuracy")
plt.plot(epochs, val_accuracy, label="validation  accuracy")
plt.legend()
plt.show()


##

