# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import serial

#####ニューラルネット#####
hw = {"height":16, "width":16}        #画像サイズ　リストではなく辞書型 中かっこで囲む
def TestProcess(imgname):
    modelname_text = open("./keras/model.json").read()
    json_strings = modelname_text.split('##########')
    textlist = json_strings[1].replace("[", "").replace("]", "").replace("\'", "").split()
    model = model_from_json(json_strings[0])
    model.load_weights("./keras/last.hdf5")  # best.hdf5 で損失最小のパラメータを使用
    img = load_img(imgname, target_size=(hw["height"], hw["width"]))    
    TEST = img_to_array(img) / 255

    pred = model.predict(np.array([TEST]), batch_size=1, verbose=0)
    print(">> 計算結果↓\n" + str(pred))
    print(">> この画像は「" + textlist[np.argmax(pred)].replace(",", "") + "」です。")
##########

#####serial通信の設定#####
#COMポートを開く
print("Open Port")
ser = serial.Serial("/dev/cu.usbserial-1410", 9600)
##########

#####ビデオ画像表示#####
#VideoCapture オブジェクトを取得
capture = cv2.VideoCapture(1)

while(True):
    ret, frame = capture.read()

    #windowのサイズを変更
    windowsize = (800, 600)
    frame = cv2.resize(frame, windowsize)

    '''
    #笑顔認識の確認
    face_cascade = cv2.CascadeClassifier("./opencv-master/data/haarcascades/haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier('./opencv-master/data/haarcascades/haarcascade_smile.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #GRAYに変換
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5) #顔と認識した矩形領域の左上のx,y座標と幅w, 高さhのリスト
    for (x,y,w,h) in faces:
        cv2.circle(frame, center = (int(x+w/2), int(y+h/2)), radius = int(w/2), color = (255, 0, 0), thickness = 2)
        faces_gray = gray[y:y+h, x:x+w] #GRAY画像から，顔領域を切り出す．
        smiles= smile_cascade.detectMultiScale(faces_gray, scaleFactor = 1.2, minNeighbors = 100, minSize = (20, 20))#笑顔識別
        if len(smiles) >0 :
            for(sx,sy,sw,sh) in smiles:
                cv2.circle(frame,(int(x+sx+sw/2),int(y+sy+sh/2)),int(sw/2),(0, 0, 255),2)#red
    '''
    
    #windowの表示
    cv2.imshow("openCV", frame)
    
    #escを押したらbreak
    k = cv2.waitKey(1)
    if k == 27: #esc
        break
    elif k == ord('s'): #s
        cv2.imwrite("../photo/smile/check.jpg", frame)
        
capture.release()
cv2.destroyAllWindows()
##########

#####撮った写真の処理#####
img = cv2.imread("../photo/smile/check.jpg", -1) #撮った写真を読み込む
cv2.imshow("origianal_image", img) #表示
TestProcess("./dataset/good_hair/0.jpg")

#####笑顔認識#####
face_cascade = cv2.CascadeClassifier("./opencv-master/data/haarcascades/haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier('./opencv-master/data/haarcascades/haarcascade_smile.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #GRAYに変換
faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5) #顔と認識した矩形領域の左上のx,y座標と幅w, 高さhのリスト
for (x,y,w,h) in faces:
                cv2.circle(img, center = (int(x+w/2), int(y+h/2)), radius = int(w/2), color = (255, 0, 0), thickness = 2) # red
                faces_gray = gray[y:y+h, x:x+w] #GRAY画像から，顔領域を切り出す．
                smiles = smile_cascade.detectMultiScale(faces_gray, scaleFactor = 1.2, minNeighbors = 100, minSize = (20, 20))#笑顔識別
                if len(smiles) >0 :
                        for(sx,sy,sw,sh) in smiles:
                                cv2.circle(img,(int(x+sx+sw/2),int(y+sy+sh/2)),int(sw/2),(0, 0, 255),2)#red

#cv2.imshow('img',img)
##########

#####笑顔認識されたらモータが回転するようにserial通信する(関数化できる？)#####
if len(smiles) > 0:
    for i in range(10): #反復回数でモーターの回転時間を変更できる
        #Aruduinoに送信
        ser.write(b"1")        
        #Aruduinoから受信
        #data = ser.read_all()
        #print("data:{}".format(data))
        time.sleep(1)
 
ser.write(b"0") #モーターを止めるよう指示
print("Close Port")
ser.close()
##########

#####
cv2.imshow('img',img)
cv2.waitKey(0) #key操作があるまで待つ
cv2.destroyAllWindows()

