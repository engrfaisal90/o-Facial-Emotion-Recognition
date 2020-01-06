from matplotlib.pyplot import imread, imshow, show, subplot, title, get_cmap, hist
from skimage.exposure import equalize_hist
import numpy as np


img = imread('C:/Users/Faisal/PycharmProjects/EmotionsGUI/last_traintest/train/5/1_AF17SUS.JPG')
eq = np.asarray(equalize_hist(img) * 255, dtype='uint8')

subplot(221); imshow(img, cmap=get_cmap('gray')); title('Original')
subplot(223); hist(img.flatten(), 256, range=(0,256));
subplot(222); imshow(eq, cmap=get_cmap('gray'));  title('Histogram Equalized')
subplot(224); hist(eq.flatten(), 256, range=(0,256));

show()