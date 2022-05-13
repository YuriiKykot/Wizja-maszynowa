# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:43:20 2022

@author: yuriy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def show_img(img, bw = False):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if bw:
        ax.imshow(img, cmap='Greys_r')
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

#Metoda dla wszyszukiwania matchow sposrod zdjecia i wideo
# za pomoca FLANN               
def FLAN2(img1, video):
    
    #Prszypisanie wideo do parametrow captured i img2
    captured, img2 = video.read()
    
    #Wyswietlanie zdjecia ktore bylo wziete do porownania
    show_img(img1)
    
    #sift - stworzenie metody dla wyszukiwania lokalnych cech w obrazach 
    # korzystalem z sift
    sift = cv2.SIFT_create()
    
    #while ktory bedzie pracowal poki sa klatki wideo
    while captured:
        #try dla tego zeby zlapac exception kiedy wideo konczy sie
        try:
            captured, img2 = video.read()
            
            #Resize klatki wideo dla łatwiejszej pracy uzytkownika
            # i dla szybszej pracy programu 
            img2 = cv2.resize(img2,(600,600))
            
            #kolorowanie zdjęć na szary
            grey1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
            grey2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
            
            #Obliczanie deskryptorow dla zdjec
            kp1, des1 = sift.detectAndCompute(grey1, None)
            kp2, des2 = sift.detectAndCompute(grey2, None)


            #Dopasowywanie wektorów deskryptorów za pomocą FLANN
            #parametry dla FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            
            #Stworzenie objekru FLANN
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            #Dopasowywanie deskryptorow
            matches = flann.knnMatch(des1,des2,k=2)
        
            #Filtrowanie dopasowan za pomocą "Lowe's ratio test"
            good = []
            for m,n in matches :
                if m.distance < 0.7*n.distance :
                    good.append(m)
                    
            #Ustawiamy warunek, że ma być co najmniej 10 dopasowań aby znaleźć obiekt,
            # w przeciwnym wypadku pokaże komunikat informujący, że nie ma takiej liczby dopasowań
            MIN_MATCH_COUNT = 10
            
            #Jeśli zostanie znaleziona wystarczająca liczba, wyodrębniamy
            # lokalizacje dopasowanych punktów kluczowych na obu obrazach
            # rysowanie prostokąta
            if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                h,w,d = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            else:
                print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
                matchesMask = None
            
            # Rysujemy nasze inliers
            draw_params = dict(matchColor = (0,255,0), # rysowanie matchow na zielono
                               singlePointColor = None,
                               matchesMask = matchesMask, # rysowanie tylko inliers
                               flags = 2)
            
            #Przypisywanie tego do innego zdjecia
            img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
            
            #Wyswietlanie zdjecia
            show_img(img3, True)
        except Exception:
            print("The End")

def main(args):    
    video = cv2.VideoCapture(args.wideo) #zapisywania wideo
    img1 = cv2.imread(args.img) #zapisywania zdjecia
    
    #sprawdzanie czy nie sa pustymi img1 i video
    #jezeli jakis z elementow jest pusty - zamyka program
    if img1 is None:
        print("img1 is None")
        sys.exit()
    if video is None:
        print("img2 is None")
        sys.exit()
    
    #Resize zdjecia dla łatwiejszej pracy uzytkownika
    # i dla szybszej pracy programu 
    img1 = cv2.resize(img1,(600,600))
    
    #przypisywanie zdjecia i wideo do metody
    FLAN2(img1,video)
    
#Metoda do pobrania wideo i zdjecia
#-i --img1 zdjecie
#-w --wideo wideo 
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--img',default=0)
    parser.add_argument('-w','--wideo',default=0)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())