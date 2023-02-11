
"""
### Saat program dijalankan akan otomatis membuat folder dalam data absen bernama hari tanggal bulan tahun pada hari tersebut. Kemudian kamera akan menyala dan akan muncul window/frame scan kartu identitas dan scan perlengkapan k3. Kengkapan k3, jika tidak memakai maka akan muncul suara peringatan.
"""

from keras.models import load_model
import time, math
import cv2 as cv
import numpy as np
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
import os
from base64 import decode
from pyzbar.pyzbar import decode
#Special for CNN
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import math
import winsound
today = (time.strftime('%a-%b-%m-%d-%Y_%H;%M;%S'))

cap = cv.VideoCapture(3)
cop = cv.VideoCapture(2)

classs ='classes.txt'
classes=[]
with open(classs,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
whT=320
confidenceth=0.5
nmsth=0.1

modelconfig='yolov3-tiny.cfg'
modelweight='yolov3_training_final.weights'
net=cv.dnn.readNetFromDarknet(modelconfig,modelweight)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv.FONT_HERSHEY_PLAIN
#nama_folder = time.strftime('%a-%d-%b-%Y')
#p = 'C:/Users/Asus/FINAL PROJECT/Data Absen/ ' + nama_folder
#os.mkdir(p)

#os.mkdir(r'C:\Users\ASUS\FINAL PROJECT\Data Absen\ '+ nama_folder)
#path = p + '/'

def findObjects(outputs,img):
   
    height, width, channels = img.shape
    bbox=[]
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[3] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.rectangle(img, (x, y), (x + w, y + 3), color, -1)
            cv.putText(img, label + " " + str(round(confidence, 2)), (x, y + 2), font, 2, (255,255,255), 1)
            cv.imwrite(path+today+'_'+nomer_anggota+'.jpg',img)
            cv.waitKey(3)
            winsound.PlaySound('pb.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
        else: 
            winsound.PlaySound('warn.wav', winsound.SND_FILENAME | winsound.SND_ASYNC )
def deteksikun(a,b):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 
    
    #deteksi helm dan warna helm
    k_lower = np.array(a, np.uint8)
    k_upper = np.array(b, np.uint8)
    k_mask = cv.inRange(hsv, k_lower, k_upper)
    kernal = np.ones((5, 5), "uint8")
    k_mask = cv.dilate(k_mask, kernal)   
    contours, hierarchy = cv.findContours(k_mask,cv.RETR_TREE,    cv.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if(area >500):
            x,y,w,h = cv.boundingRect(contour)
            cv.putText(img, 'pekerja', (20,20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_4)
            cv.putText(img, 'NIS: '+nomer_anggota, (20,55), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv.LINE_4)
            
nomer_anggota = ''           
while True:
    _, img= cap.read()
    #barcode
    _, frame = cop.read() 
   
    imgi = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(frame, 0)
    cv.putText(frame, 'Pindai Kartu Identitas', (450,450), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_4)
    cv.putText(img, 'Silahkan Pakai Perlengkapan K3 ', (45,430), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_4)
    colored_negative = abs(255-imgi)
    gray_negative = abs(255-gray)

    e = 0
    deteksikun([30, 100, 100],[50, 255, 255])
    for cartbarcode in decode(gray_negative):
        myData1 = cartbarcode.data.decode('utf-8')

        ptscart = np.array([cartbarcode.polygon],np.int32)
        ptscart = ptscart.reshape((-1,1,2))
        cv.polylines(frame,[ptscart],True,(255,0,255),5)
        pts2 = cartbarcode.rect
        winsound.Beep(1000, 700)
        cv.putText(frame,myData1,(pts2[0],pts2[1]),cv.FONT_HERSHEY_SIMPLEX, 0.9,(255,0,255),2) 
        nomer_anggota = myData1
        blob=cv.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
    
        
        findObjects(outs,img)
   
    cv.imshow('scan kartu',frame) 
    cv.imshow('scan k3', img)
    key=cv.waitKey(1)
    if key==27:
        break
        
cap.release()
cop.release()
cv.destroyAllWindows()  
