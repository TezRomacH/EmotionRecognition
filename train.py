# coding=utf-8
import cv2
import glob
import random
import numpy as np
import pylab

import sys
import time


class Profiler:
    def __init__(self):
        self._startTime = time.time()

    def printTime(self):
        print("Time: {:.3f} sec".format(time.time() - self._startTime))

emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]
emotion_dict = {0: "neutral", 1: "anger", 3: "disgust", 4: "fear", 5: "happy", 6: "sadness", 7: "surprise"}
faceRecognizer = cv2.face.createFisherFaceRecognizer()

cap = cv2.VideoCapture(0)

GET_ALL_FILES = True


def getTrainingPredictionFiles(emotion):
    files = glob.glob("dataset\\%s\\*" % emotion)
    random.shuffle(files)
    prediction = []
    if GET_ALL_FILES:
        training = files
    else:
        training = files[:int(len(files) * 0.8)]  # первые 80% - это для тренировки
        prediction = files[-int(len(files) * 0.2):]  # остальные 20% - это для предсказаний

    return training, prediction


def makeTrainSets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:
        training, prediction = getTrainingPredictionFiles(emotion)

        # Присоеденияет данные к наору
        for item in training:
            image = cv2.imread(item)  # открывает изображение
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # переводит изображение в граждации серого
            training_data.append(gray)

            label = emotions.index(emotion)
            correct_label = label if label < 2 else label + 1
            training_labels.append(correct_label)

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


def trainFaceClassifier():
    training_data, training_labels, prediction_data, prediction_labels = makeTrainSets()
    print "Размер тренировочного набора - " + str(len(training_labels)) + " изображений"
    faceRecognizer.train(training_data, np.asarray(training_labels))


def detect_face(filename):
    files = glob.glob(filename)
    for f in files:
        frame = cv2.imread(f)
        return detectFaceFromImage(frame)

def detectFaceFromImage(image):
    faceDet1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Переводим изображение в градации серого

    # Находим лицо четыремя разнамы классификаторами
    face1 = faceDet1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # Проходим по всем найденным лицам и выбираем первое
    if len(face1) == 1:
        facefeatures = face1
    elif len(face2) == 1:
        facefeatures = face2
    elif len(face3) == 1:
        facefeatures = face3
    elif len(face4) == 1:
        facefeatures = face4
    else:
        facefeatures = ""

    returned_path = ""

    for (x, y, w, h) in facefeatures:  # Получаем прямоуголькик с лицом
        gray = gray[y:y + h, x:x + w]  # Вырезаем лицо

        try:
            out = cv2.resize(gray, (350, 350))  # Делаем изображение одного размера с нашим набором
            count_files = len(glob.glob('cache\\*'))
            returned_path = "cache\\%s.jpg" % count_files
            cv2.imwrite(returned_path, out)
            returned_path = "cache/%s.jpg" % count_files
        except:
            print "ошибка!"
            pass  # В случае ошибки, пропускаем файл

    return returned_path

faceRecognizer.load("trained.bin")  # Загружаем faceRecognizer

p = Profiler()
if sys.argv.__contains__("-c") or sys.argv.__contains__("--camera"):
    r, im = cap.read()
    pylab.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    pylab.show()
    path = detectFaceFromImage(im)
else:
    # путь к картинке нам потребуется в любом случае
    if len(sys.argv) < 2:
        img_path = input("Enter the path to the file: ")
    else:
        img_path = sys.argv[1]
        print img_path

    path = detect_face(img_path)

image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
prediction_label = faceRecognizer.predict(image)[0]
p.printTime()

print "label = " + str(prediction_label)
title = "Predict = " + str(emotion_dict[prediction_label])
print title
pylab.imshow(image, cmap='gray')
pylab.title(title)
pylab.show()
