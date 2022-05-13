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
# za pomoca Brute-Force         
def BruteForce(img1, video):   
    
    #Prszypisanie wideo do parametrow captured i img2
    captured, img2 = video.read()
    
    #Wyswietlanie zdjecia ktore bylo wziete do porownania
    show_img(img1)
    
    #orb - stworzenie metody dla wyszukiwania lokalnych cech w obrazach 
    # korzystalem z orb
    orb = cv2.ORB_create()
    
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
            kp1, des1 = orb.detectAndCompute(grey1, None)
            kp2, des2 = orb.detectAndCompute(grey2, None)
        
            #Stworzenie objektu BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            #Dopasowywanie deskryptorow
            matches = bf.match(des1,des2)
            #Sortowanie ich według odległości.
            matches = sorted(matches, key = lambda x:x.distance)
            
            #Używamy 100 najlepszych dopasowań, aby utworzyć macierz transformacji
            good_matches = matches[:100]
        
            #Przekształćanie prostokąta wokół img1 na podstawie macierzy transformacji
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            matchesMask = mask.ravel().tolist()
            h,w = img1.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

            dst = cv2.perspectiveTransform(pts,M)
            dst += (w, 0) # adding offset

            draw_params = dict(matchColor = (0,255,0), # rysowanie matchow na zielono
                               singlePointColor = None,
                               matchesMask = matchesMask, # rysowanie tylko inliers
                               flags = 2)

            img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None,**draw_params)

            # Rysowanie obwiednię w kolorze czerwonym
            img3 = cv2.polylines(img3, [np.int32(dst)], True, (255,0,0), 3, cv2.LINE_AA)
            
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
    BruteForce(img1,video)
    
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