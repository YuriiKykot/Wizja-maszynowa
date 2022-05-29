# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:35:51 2022

@author: yuriy
"""

import cv2
import numpy as np
import argparse
import sys

#Metoda dla Optical Flow z Fernerback i Lucas-Kanade
def Play(video):
    #Wczytanie pierwszej klatki
    captured, old_frame = video.read()
    ################################### Ferneback
    #Konwertujemy na szary
    prsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #Utwórzamy maske
    hsv = np.zeros_like(old_frame)
    #Ustawinie nasycenia obrazu na maks
    hsv[...,1] = 255
    ################################## LK
    #parametry do wykrywania rogów ShiTomasi
    feature_params = dict( maxCorners = 100,
                          qualityLevel = 0.3,
                          minDistance = 7,
                          blockSize = 7 )
    #Parametry do przepływu optycznego Lucasa kanade
    lk_params = dict( winSize  = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #Utwórzenie losowego koloru
    color = np.random.randint(0,255,(100,3))
    
    #Birżemy pirwsza klatke
    captured, old_frame = video.read()
    #Konwertujemy na szary
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #Szukamy rogi
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    
    #Utwórzamy maskie do rysowania
    mask = np.zeros_like(old_frame)
    
    #################################### While z dwoma metodami dla wscytania wideo klatka po klatce
    while video.isOpened():
        # Bierzemy kolejną klatke
        captured,frame = video.read()
        #Zapisuje oryginalny obraz
        img1 = frame
        
        ################################################ Ferneback
        #Konwertujemy na szary klatke
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Liczenie optical flow
        flow = cv2.calcOpticalFlowFarneback(prsv, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #Obliczenie wielkośći i kąta wektora 2D
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        #Ustawinie wartośći odcienia obrazu zgodnie z kątem optical flow
        hsv[...,0] = ang * 180/np.pi/2
        #Ustawienie wartośći zgodnie ze znormalizowaną wielkością oprical flow
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Konwertujemy w rgb
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        ################################################# LK
        #Konwertujemy na szary klatke
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Liczymy optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        #Wybieramy lepsze punkty
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        
        #Rysujemy drogi
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame,mask)

        ########################################################### Wyswietlenie obrazow
        #Resize trszech obrazow (Oryginalny, Ferneback, LK)
        img1 = cv2.resize(img1,(600,600))
        img2 = cv2.resize(rgb,(600,600))
        img3 = cv2.resize(img,(600,600))
        
        #Umieszczenie ich na jednej linii
        Hori = np.concatenate((img1, img2, img3), axis=1)
        Hori = cv2.resize(Hori,(1200,600))
        #Wyświetlanie trzech obrazow na horyzontalnej linii
        cv2.imshow('Result', Hori)
        
        #Przyciski q-wylacza program, s-zapisuje obraz
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('opticalflow.png', Hori)
            
        ################################ Ferneback
        prsv = next
        ############################### LK
        
        # Aktulizujemy poprzednią klatkę i poprzednie punkty
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        
    captured.release()
        

def main(args) :    
    video = cv2.VideoCapture(args.wideo) #zapisywania wideo
    
    #sprawdzanie czy nie sa pustymi img1 i video
    #jezeli jakis z elementow jest pusty - zamyka program
    if video is None:
        print("Video is None")
        sys.exit()
    #try dla tego zeby zlapac exception kiedy wideo konczy sie
    try:
        Play(video)
    except Exception:
        print("The End")
    
    cv2.destroyAllWindows()
    
#Metoda do pobrania wideo i zdjecia
#-w --wideo wideo 
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--wideo',default=0)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())
