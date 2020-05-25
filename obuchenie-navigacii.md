# Обучение навигации

Давайте соберем нашу тренировочную программу. Создание тренировочных данных с помощью управления роботом разделяет наши данные на три набора – левый поворот, правый поворот и движение прямо. У нас есть наши обучающие изображения в трех подпапках с соответствующими названиями. Считываем эти данные, связываем их с метками и предварительно обрабатываем, чтобы представить их нейронной сети:

```text
# -*- coding: utf-8 -*-
"""
CNN based robot navigation – TRAINING program
@author: Francis Govers
"""
```

Эта программа была частично вдохновлена блогом Адриана Розброка pyImageSearch и статьей "Deep Obstacle Avoidance" Салливана и Лоусона из военно-морской исследовательской лаборатории. Вот импорты, которые потребуются для этой программы:

```text
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
```

Настроим нашу CNN. У нас есть три слоя свертки, за каждым из которых следует максимальный слой объединения. Помните, что каждый слой maxpooling уменьшит разрешение изображения, рассматриваемого сетью, наполовину, что составляет 1⁄4 данных, потому что мы уменьшаем вдвое ширину и высоту.

Слои свертки используют функцию активации ReLU, так как нам не нужны никакие отрицательные значения пикселей.

После слоев свертки идут два полностью связанных слоя по 500 нейронов в каждом. Последний слой – это наши три выходных слоя нейронов, с классификатором Softmax, который будет выводить процент каждой классификации \(слева, справа, в центре\). Результат будет выглядеть следующим образом \(0.8, 0.15, 0.05\) с тремя числами, которые складываются в 1.

Далее представлен универсальный класс сети свертки, который может быть повторно использован для других целей, так как это общая многоклассовая классификация изображений CNN:

```text
class ConvNet():
@staticmethod
def create(width, height, depth, classes):
# initialize the network
network = Sequential()
inputShape = (height, width, depth)
# first set of CONV => RELU => POOL layers
network.add(Conv2D(50, (10, 10), padding="same",
input_shape=inputShape))
network.add(Activation("relu"))
network.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# second set of CONV => RELU => POOL layers
network.add(Conv2D(50, (5, 5), padding="same"))
network.add(Activation("relu"))
network.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# third set of CONV => RELU => POOL layers
network.add(Conv2D(50, (5, 5), padding="same"))
network.add(Activation("relu"))
network.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Fully connected ReLU layers
network.add(Flatten())
network.add(Dense(500))
network.add(Activation("relu"))
network.add(Dense(500))
network.add(Activation("relu"))

# softmax classifier
network.add(Dense(classes))
network.add(Activation("softmax"))
# return the constructed network architecture
return network
```

Далее мы устанавливаем режим обучения. Мы проведем 25 тренировочных пробегов со скоростью обучения 0,001. Устанавливаем размер пакета в 32 изображения на пакет, который мы можем уменьшить, если в конечном итоге у нас закончится память.

```text
EPOCHS = 25
LEARN_RATE = 1e-3
BATCH = 32 # batch size - modify if you run out of memory
```

Этот раздел подгружает все наши изображения. Мы помещаем три типа обучающих изображений в папки, называемые левой, правой и центральной. Поместим все изображения в список под названием Images и точно так же поместим метки в одноименный список:

```text
print ("Loading Images")
images=[]
labels=[]
#location of your images
imgPath = "c:\users\fxgovers\documents\book\chapter7\train\"
imageDirs=["left","right","center"]
for imgDir in imageDirs:
fullPath = imgPath + imgDir
# find all the images in this directory
allFileNames = os.listdir(fullPath)
ifiles=[]
label = imgDirs.index(imgDir) # use the integer version of the label
# 0= left, 1 = right, 2 = center
for fname in allFileNames:
if ".jpg" in fname:
ifiles.append(fname)
```

Теперь можно вернуться к схеме процесса, который будет использоваться, чтобы предварительно обработать изображения. Разрежем изображение пополам и просто обработаем верхнюю половину картинки. Затем уменьшим изображение до 244х244, чтобы вписаться в нейронную сеть, которая нуждается в квадратных изображениях. Мы преобразуем изображение в оттенки серого \(черно-белый\), так как нам не нужно рассматривать цвет, а только формы. Это еще больше сокращает наши данные. Мы выравниваем изображение, которое масштабирует диапазон серых цветов, чтобы заполнить всю область от 0 до 255. Это выравнивает освещение и устанавливает контраст:

```text
# process all of the images
for ifname in ifiles:
# load the image, pre-process it, and store it in the data list
image = cv2.imread(ifname)
# let's get the image to a known size regardless of what was collected
image = cv2.resize(image, (800, 600))
halfImage = 800*300 # half the pixels
# cut the image in half -we take the top half
image = image[0:halfimage]
#size the image to what we want to put into the neural network
image=cv2.resize(image,(224,224))
# convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#equalize the image to use the full range from 0 to 255
# this gets rid of a lot of illumination variation
image = cv2.equalizeHist(image)
```

Далее идет размытие по Гауссу. Это необязательный элемент – его можно удалить, если ваша комната не имеет много деталей. В нашей игровой комнате много мебели, поэтому уменьшение шума улучшит производительность:

```text
# gaussian blur the image to remove high freqency noise
# we use a 5x kernel
image = cv2.GaussianBlur(img,(5,5),0)
```

Далее преобразуем изображение в массив floats NumPy от 0 до 1, вместо набора целых чисел от 0 до 255. Мы также помещаем число, связанное с метками \(left = 0, right = 1, center = 2\), в соответствующий массив меток NumPy:

```text
# convert to a numpy array
image = img_to_array(image)
# normalize the data to be from 0 to 1
image2 = np.array(image, dtype="float") / 255.0
images=images.append(image)
labels.append(label)
labels = np.array(labels) # convert to array
```

Разделяем данные на две части – обучающую выборку, которую мы используем для обучения нейронной сети, и тестовую выборку, с помощью которой мы проверяем обучающую. Мы будем использовать 80% образцов изображений для обучения и 20% для тестирования:

```text
# split data into testing data and training data 80/20
(trainData, testData, trainLabel, testLabel) = train_test_split(data,
labels, test_size=0.20, random_state=42)
```

Преобразуем метки в тензор, который является просто определенным форматом данных:

```text
# convert the labels from integers to vectors
trainLabel = to_categorical(trainLabel, num_classes=3)
testLabel = to_categorical(testLabel, num_classes=3)
```

Теперь строим настоящую нейронную сеть, создавая экземпляр объекта ConvNet, который фактически строит наш CNN в Keras. Мы настроили оптимизатор ADAM, который является типом адаптивного градиентного спуска. ADAM выступает за адаптивную оценку момента. Он действует против градиента погрешности, как тяжелый шар с трением, – он имеет некоторый импульс, но не набирает скорость быстро:

```text
# initialize the artificial neural network
print("compiling CNN...")
cnn = ConvNet.build(width=224, height=224, depth=1, classes=3)
opt = Adam(lr=LEARN_RATE, decay=LEARN_RATE / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])
```

На этом шаге происходит обучение нейросети. Это займет довольно много времени. Нацелимся на 1000 изображений в каждом наборе, что составляет примерно 50 минут езды робота:

```text
# train the network
print("Training network. This will take a while")
trainedNetwork = model.fit_generator(aug.flow(trainImage, trainLabel,
batch_size=BATCH),
validation_data=(testImage, testLable), steps_per_epoch=len(trainImage) //
BATCH,
epochs=EPOCHS, verbose=1)
# save the model to disk
print("Writing network to disk")
cnn.save("nav_model")
```

Все готово! Теперь сохраняем созданную модель на диск, чтобы перенести ее на Raspberry Pi.

Теперь сделайте второй тренировочный набор движения от случайных мест до коробки игрушек. Выберите случайные места и используйте джойстик, чтобы привести робота к коробке игрушек из каждого. Продолжайте идти, пока у вас не будет 1000 изображений или около того. Запустите их через обучающую программу и обозначьте эту модель toy box\_model, изменив последнюю строку программы:

```text
cnn.save(“toybox_model”)
```



