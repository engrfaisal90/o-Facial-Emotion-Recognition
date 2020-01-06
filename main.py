import sys

import numpy as np
import cv2
import PyQt5
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
import face_extraction as fe
import dlib
from keras.models import load_model
import keras

class UserInterface(QDialog):
    def __init__(self):
        self.face6=[]
        self.itera=0
        self.flag=False
        self.emotion=""
        self.predictor_path = "code\preprocessing\shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.model = load_model("code\Training\cnn_model_keras_newdataset_newnet_new_200.h5")
        #self.model = load_model("code\Training\my_last_cnn_model_keras_newdataset_newnet.h5")
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        super(UserInterface, self).__init__()
        loadUi('userInterface.ui', self)
        self.videoLabel.setText("Select Image or Video mode")
        self.image=None
        self.face=None
        self.probLabel.setText(" ")
        self.imgButton.clicked.connect(self.img_mood)
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.vidButton.clicked.connect(self.vid_mood)
        self.stopButton.hide()
        self.startButton.hide()


    def start_webcam(self):
        self.emoLabel.setText(" ")
        self.emotion=" "
        self.itera=0
        self.stopButton.setEnabled(True)
        self.startButton.setEnabled(False)
        self.vidButton.setEnabled(False)
        self.imgButton.setEnabled(False)
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.timer = QTimer(self)

        self.flag=True
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image=cv2.flip(self.image, 1)
        self.display_image(self.image, 1)
        try:
            self.face=self.extract_image(self.image)
            if self.face is not None:
                    self.predict_emotion(self.face)
        except:
            set.cropLabel.setText("Many Faces.\nSystem is \ndesigned for\nsingle face")


    def display_image(self, img, windo):
        qformat = QImage.Format_Indexed8
        if windo == 1:
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888

            out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
            out_image = out_image.rgbSwapped()

            self.videoLabel.setPixmap(QPixmap.fromImage(out_image))
            self.videoLabel.setScaledContents(True)
        else:
            cv2.resize(img, (80, 100))
            qformat=QImage.Format_Grayscale8
            out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
            out_image = out_image.rgbSwapped()
            self.cropLabel.setPixmap(QPixmap.fromImage(out_image))
            self.cropLabel.setScaledContents(True)

    def extract_image(self, img1):

        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # print(gray.shape)
        try:
            img2 = fe.preprocessing(gray, self.predictor)

            if len(img2)>0:
                self.display_image(img2[0],2)
                return img2[0]
            else:
                self.cropLabel.setText("NO face Detected")
                return None
        except:
            self.cropLabel.setText("multi faces")
            self.emoLabel.setText("")
            return None

    def predict_emotion(self, ext):

        im1, im2, im3, im4= fe.intensity_norm(ext)
        im1 = cv2.resize(im1, (80, 100))
        im2 = cv2.resize(im2, (80, 100))
        im3 = cv2.resize(im3, (80, 100))
        im4 = cv2.resize(im4, (80, 100))
        #im5 = cv2.resize(im5, (100, 100))
        self.face6.append(im1)
        self.face6.append(im2)
        self.face6.append(im3)
        self.face6.append(im4)
        #self.face6.append(im5)

        try:
            if self.itera%2==0:
                imgs = np.reshape(self.face6, (len(self.face6), 80, 100, 1))
                pred_k = self.model.predict(imgs)

                add=pred_k.sum(axis=0)/len(pred_k)
                add=np.round(add,3)
                self.probLabel.setText('Angry:'+str(add[0])+"   Disgust:"+str(add[1])+" Afraid:"+str(add[2])+"\nHappy:"+str(add[3])+"   Sad:"+str(add[4])+"    Surprise:"+str(add[5])+"\nNeutral:"+str(add[6]))
                emodict = {0: "Angry", 1: "Disgust", 2: "Afraid", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
                index = add.argmax()
                emot = emodict[index]
                self.emotion=emot+"\n"+self.emotion
                self.emoLabel.setText(emot)
            else:
                self.itera = self.itera + 1
                return 0
        except:
            self.emoLabel.setText("error")
            self.itera = self.itera + 1

    def img_mood(self):
        self.itera=2
        self.emoLabel.setText("")
        self.probLabel.setText(" ")
        self.emotion=""
        if self.flag==True:
            self.timer.stop()
            self.capture.release()
        self.startButton.hide()
        self.stopButton.hide()
        self.vidButton.setEnabled(True)
        self.emoLabel.setText("")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All files (*.*);; PNG files (*.png);; TIFF files (*.tiff);; JPG files (*.jpg)", options=options)
        if fileName:
            self.image=cv2.imread(fileName)
            self.display_image(self.image,1)
            self.face = self.extract_image(self.image)
            if self.face is not None:
                self.predict_emotion(self.face)
            else:
                self.emoLabel.setText("No face found")




    def vid_mood(self):
        self.itera=0
        self.emoLabel.setText("")
        self.probLabel.setText(" ")
        self.stopButton.show()
        self.startButton.show()
        self.stopButton.setEnabled(False)
        self.vidButton.setEnabled(False)
        self.emotion=""





    def stop_webcam(self):
        self.flag=False
        self.cropLabel.setText("no stream")
        self.stopButton.setEnabled(False)
        self.startButton.setEnabled(True)
        self.vidButton.setEnabled(False)
        self.imgButton.setEnabled(True)
        self.timer.stop()
        self.capture.release()
        self.videoLabel.setText("Streaming stopped \nSelect Image or Video mode")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UserInterface()
    window.setWindowTitle('Emotions Detection')
    window.show()
    sys.exit(app.exec_())
