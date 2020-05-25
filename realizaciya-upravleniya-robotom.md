# Реализация управления роботом

Отлично – мы построили и обучили нашу нейросеть. Теперь нужно использовать ее, чтобы управлять роботом. Нужно объединить программу, которая посылает команды, с нашим процессом классификации нейронной сети. Я добавил некоторые команды через раздел Ros syscommand, который используется для непериодических команд для роботов. Команда Sys просто публикует строку, так что вы можете использовать ее практически для чего угодно:

```text
# -*- coding: utf-8 -*-
"""
ROS Neural Network based Navigation Program
@author: Francis Govers
"""
# navigation program
# using neural network with ROS interface
```

Начнем с импорта из ROS, из OpenCV 2 и из Keras, так как мы будем объединять функции из всех трех библиотек:

```text
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeErrorfrom keras.preprocessing.image
import img_to_array
from keras.models import load_model
import numpy as np
```

В первом разделе находится интерфейс ROS. Мне нравится инкапсулировать интерфейс ROS таким образом, чтобы все публикации и подписки находились в одном месте. У нас есть несколько пунктов для настройки – нам нужно иметь возможность отправлять и получать команды по syscommand. Публикуем команды для двигателей робота в теме cmd\_vel. Получаем изображения с камеры на image\_topic. Используем обратные вызовы для обработки события, когда тема публикуется в другом месте робота. Они могут быть вызваны в любое время. Больше контроля обеспечивается через часть, которая обрабатывается с помощью методов pubTwist и pubmed. Добавим флаги к полученным командам и изображениям, чтобы случайно не обрабатывать одно и то же изображение или команду дважды:

```text
class ROSIF():
def __init__(self):
self.bridge = CvBridge()
self.image_sub = rospy.Subscriber("image_topic",Image,self.callback)
self.cmd_sub = rospy.Subscriber("syscommand",String,self.cmdCallback)
self.cmd_pub = rospy.Publisher("syscommand",String,queue_size=10)
self.twist_pub = rospy.Publisher("cmd_vel",Twist,queue_size=10)
self.newImage = False
self.cmdReceived=""

def callback(self):
try:
self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
self.newImage = True
except CvBridgeError as e:
print(e)
def cmdCallback(self,data):
# receieve a message on syscommand
self.cmdReceived = data.data

def getCmd(self):
cmd = self.cmdReceived
self.cmdReceived = "" # clear the command so we dont do it twice
return cmd
```

Эта функция помогает остальной части программы получить последнее изображение из камеры, которая публикуется на ROS на image\_topic. Мы берем последнее изображение и устанавливаем новую переменную изображения в False, чтобы в следующий раз знать, пытаемся ли мы обработать одно и то же изображение дважды. Каждый раз, когда мы получаем новый образ, мы устанавливаем newImage в Тrue, и каждый раз, когда мы используем изображения, мы переводим newImage значение False.

```text
def getImage(self):
if self.newImage=True:
self.newImage = False # reset the flag so we don't process twice
return self.image
self.newImage = False
# we send back a list with zero elements
img = []
return img
```

Следующий раздел посылает роботу команды скорости, чтобы соответствовать тому, что предсказывает нам выход CNN. Выходные данные нейросети – это одно из трех значений: влево, вправо или вперед. Они выходят из нейронной сети в виде одного из трех перечисленных значений – 1, 2 или 3. Мы преобразуем их обратно в левое, правое и центральное значения, а затем используем эту информацию для отправки команды движения роботу. Робот использует твист – сообщение для отправки моторных команд. Сообщение данных Twist предназначено для размещения очень сложных роботов, квадрокоптеров и систем полного привода, которые могут двигаться в любом направлении, поэтому оно имеет много дополнительных значений. Даем команду Twist.linear.x для установки скорости робота вперед и назад, а также Twist.angular.z для установки поворота или перемещения основания. В нашем случае положительное значение angular.z ведет вправо, а отрицательное значение – влево. Наш последний оператор публикует значения данных в теме cmd\_vel в виде твист-сообщения.

```text
# publishing commands back to the robot
def pubCmd(self,cmdstr):
self.cmd_pub.publish(String(cmdstr)):
def pubTwist(self,cmd):
if cmd == 0: # turn left
turn = -2
speed = 1
if cmd==1:
turn = 2
speed = 1
if cmd ==3:
turn=0
speed = 1
# all stop
if cmd==4:
turn = 0
speed = 0
cmdTwist = Twist()
cmdTwist.linear.x = speed
cmdTwist.angular.z = turn
self.twist_pub.publish(cmdTwist)
```

Далее создадим функцию для выполнения всей нашей обработки изображений с помощью одной команды. Это точная копия того, как мы предварительно обработали изображения для тренировочной программы. Может показаться немного странным, что мы увеличиваем изображение только для того, чтобы затем снова уменьшить его. Причина этого заключается в необходимости иметь детализацию для вертикальной части изображения. Если бы мы уменьшили его до 240x240, а затем разрезали пополам, потом мы бы растягивал пиксели, чтобы снова сделать его квадратным. Лучше иметь дополнительные пиксели при уменьшении масштаба. Большое преимущество этой техники заключается в том, что не имеет значения, в каком разрешении находится входящее изображение – в конечном итоге мы получим правильно подобранное и обрезанное изображение.

Следующие шаги – перевод изображения в оттенки серого, выравнивание в диапазоне контрастности, которое расширяет значения цвета, чтобы заполнить доступное пространство, и выполнение размытия по Гауссу для уменьшения шума. Нормализуем изображение для нейронной сети путем преобразования наших целочисленных значений 0-255 оттенков серого в значения с плавающей точкой от 0 до 1:

```text
def processImage(img):
# need to process the image
image = cv2.resize(image, (640, 480))
halfImage = 640*240 # half the pixels
# cut the image in half -we take the top half
image = image[0:halfimage]
#size the image to what we want to put into the neural network
image=cv2.resize(image,(224,224))
# convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#equalize the image to use the full range from 0 to 255
# this gets rid of a lot of illumination variation
image = cv2.equalizeHist(image)
# gaussian blur the image to remove high freqency noise
# we use a 5x kernel
image = cv2.GaussianBlur(img,(5,5),0)
# convert to a numpy array
image = img_to_array(image)
# normalize the data to be from 0 to 1
image2 = np.array(image, dtype="float") / 255.0
return image2
```

Теперь перейдем к основной программе. Нужно инициализировать узел ROS, чтобы мы могли общаться с системой публикации/подписки ROS. Мы создаем переменную mode, которая используется для определения, какая ветвь обработки должна идти вниз. Создаем интерфейс, позволяющий оператору включать и выключать навигационную функцию, а также выбирать между обычной навигацией и нашим режимом поиска игрушечной коробки.

В первом разделе загружаем обе модели нейронных сетей, которые мы обучали ранее:

```text
# MAIN PROGRAM
ic = image_converter()
rosif = ROSIF()
rospy.init_node('ROS_cnn_nav')
mode = "OFF"
# load the model for regular navigation
navModel = load_model("nav_model")
toyboxModel = load_model("toybox_model")
```

Далее начинается цикл обработки, который выполняется во время работы программы. В процессе работы rospy.spin\(\) сообщает системе ROS, что требуется обработать любое сообщение, которое ждет нас. Последний шаг состоит в том, чтобы приостановить программу на 0,02 секунды, чтобы позволить Raspberry Pi обрабатывать другие данные и запускать другие программы:

```text
while not rospy.is_shutdown():
rospy.spin()
time.sleep(0.02)
```



