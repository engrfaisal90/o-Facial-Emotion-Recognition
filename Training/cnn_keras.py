import numpy as np
import pickle
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib as plt
K.set_image_dim_ordering('tf')



image_x, image_y = 80,100 #Image size i.e rows and cols




def cnn_model():
	#construct CNN structure
    num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral

    model = Sequential()
    
    #1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(image_x,image_y,1)))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
    
    #2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
    
    #3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
    
    
    model.add(Flatten())
    
    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_classes, activation='softmax'))
    sgd = optimizers.SGD(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath="cnn_model_keras_newdataset_newnet.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint2 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint1, checkpoint2]
    return model, callbacks_list

def train():
    os.chdir('../..')
    with open("pickledata/nd_train_data", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("pickledata/nd_train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)
    with open("pickledata/nd_test_data", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("pickledata/nd_test_label", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))
    test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)
    global scores
	

    model, callbacks_list = cnn_model()
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=100, batch_size=100, callbacks=callbacks_list)
    scores = model.evaluate(test_images, test_labels, verbose=0)
    #pred_keras = model.predict(test_images).ravel()
    #fpr_k, tpr_k, thr_k = roc_curve(test_labels, pred_keras)
    #auc_k = auc(fpr_k, tpr_k)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))
    #roc_print(fpr_k,tpr_k,auc_k)
    model.save('cnn_model_keras_newdataset_newnet.h5')
   

train()
