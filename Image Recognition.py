# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:17:10 2020

@author: user
"""

from PIL import Image
import os
import numpy as np
np.random.seed(10)
from random import shuffle
from tqdm import tqdm
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
# # 資料準備


TRAIN_DIR = 'TRAIN'
IMG_SIZE = 32


def label_img(img):
    # Images are formatted as: ADIDAS_1, NIKE_3 ...
    word_label = img.split('_')[0]
    if word_label == 'NIKE': return 1 #one hot encoding
    elif word_label == 'ADIDAS': return 0 #one hot encoding


def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        ############################################################
        #    This part is different from sentdex's tutorial
        # Chose to use PIL instead of cv2 for image pre-processing
        ############################################################
        
        img = Image.open(path) #Read image syntax with PIL Library
        #img = img.convert('L') #Grayscale conversion with PIL library
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS) #Resizing image syntax with PIL Library
        
        ############################################################
        
        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    
    return train_data


train_data = create_train_data()
train = train_data #Train Set

test = train_data[:-120] #Validation Set
train_x = np.array([i[0] for i in train])
train_y = [i[1] for i in train]

test_x = np.array([i[0] for i in test])
test_y = [i[1] for i in test]

x_img_train_normalize = train_x.astype('float32') / 255.0
x_img_test_normalize = test_x.astype('float32') / 255.0

y_label_train_OneHot = np_utils.to_categorical(train_y)
y_label_test_OneHot = np_utils.to_categorical(test_y)

model = Sequential()


#padding：補零方式，卷積層取週邊kernel_size的滑動視窗時，若超越邊界時，是否要放棄這個output點(valid)、一律補零(same)、還是不計算超越邊界的Input值(causal)。
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32, 32, 4), 
                 activation='relu', 
                 padding='same'))



model.add(Dropout(rate=0.25))



model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))


model.add(Dropout(0.25))


model.add(MaxPooling2D(pool_size=(2, 2)))


#Step3	建立神經網路(平坦層、隱藏層、輸出層)



model.add(Flatten())
model.add(Dropout(rate=0.25))


model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(2, activation='softmax'))

print(model.summary())

# # 訓練模型

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#verbose:0 安靜 、 1 更新信息
train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,validation_split=0.2,epochs=100, batch_size=128, verbose=1)          



def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


show_train_history('acc','val_acc')


show_train_history('loss','val_loss')


# # 評估模型準確率

scores = model.evaluate(x_img_test_normalize, 
                        y_label_test_OneHot, verbose=0)

# # 進行預測

prediction=model.predict_classes(x_img_test_normalize)


# # 查看預測結果

label_dict={0:"ADIDAS",1:"NIKE"}


def plot_images_labels_prediction(images,labels,prediction,
                                  idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(i)+' , '+label_dict[labels[i]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()


test_y=np.array(test_y)
#test_x=test_x.reshape(-1,IMG_SIZE,IMG_SIZE)
plot_images_labels_prediction(test_x,test_y,
                              prediction,0,10)

# # 查看預測機率

Predicted_Probability=model.predict(x_img_test_normalize)


def show_Predicted_Probability(y,prediction,
                               x_img,Predicted_Probability,i):
    print('label:',label_dict[y[i]],
          'predict:',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(test_x[i],(32, 32, 4)))
    plt.show()
    for j in range(2): #字典數量
        print(label_dict[j]+' Probability:%1.9f'%(Predicted_Probability[i][j]))



show_Predicted_Probability(test_y,prediction,
                           test_x,Predicted_Probability,0)


show_Predicted_Probability(test_y,prediction,
                           test_x,Predicted_Probability,3)

