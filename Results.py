# coding: utf-8

# In[5]:

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib as plt
from keras.models import load_model
import os

# In[ ]:

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('D:/Educational Data/tigp phd data/Projects/MM Project/my_last_cnn_model_keras_newdataset_newnet.h5')

# In[ ]:

import pickle
import numpy as np

with open("D:/Educational Data/tigp phd data/Projects/MM Project/pickledata/my_last_nd_test_data", "rb") as f:
    test_images = np.array(pickle.load(f))
with open("D:/Educational Data/tigp phd data/Projects/MM Project/pickledata/my_last_nd_test_label", "rb") as f:
    test_labels = np.array(pickle.load(f), dtype=np.int32)

# In[ ]:

size = test_labels.size

# In[ ]:

size

# In[7]:

imgs = np.reshape(test_images, (size, 100, 100, 1))
pred_k = model.predict_classes(imgs)

# In[ ]:


# In[13]:

from sklearn.preprocessing import label_binarize

pred_k_b = label_binarize(pred_k, classes=[0, 1, 2, 3, 4, 5, 6])
test_b = label_binarize(test_labels, classes=[0, 1, 2, 3, 4, 5, 6])

# In[15]:

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(7):
    fpr[i], tpr[i], _ = roc_curve(test_b[:, i], pred_k_b[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# In[16]:
from matplotlib import pyplot as plt

# In[18]:

emotion = ["Angry", "Digust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
for i in range(7):

    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    if i == 33:
        plt.xlim([0.0, 1.0])
        plt.ylim([0.9, 1.05])
    else:

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Cruve for Emotion ' + str(emotion[i]))
    plt.legend(loc="lower right")
    # plt.savefig('D:/tmp/out/%d.jpg'%i)
    plt.show()

# In[19]:

from sklearn.metrics import classification_report

y_true = list(test_labels)
y_pred = list(pred_k)
target_names = ['0', '1', '2', '3', '4', '5', '6']
print(classification_report(y_true, y_pred, target_names=target_names))

# In[ ]:


# In[23]:

import matplotlib.pyplot as plt

plt.plot(pred_k)
plt.plot(test_labels)

plt.xlim([0, 100])
plt.ylim([0, 10])

plt.legend(['predicted', 'ground truth'], loc='upper left')

plt.show()

# In[ ]:


# In[ ]:



