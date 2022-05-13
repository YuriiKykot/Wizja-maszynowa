# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 21:45:46 2022

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

#Metoda dla wszyszukiwania matchow sposrod dwoch zdjec
# za pomoca Brute-Force            
def BruteForce(img1, img2):
    
    #kolorowanie zdjęć na szary
    grey1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    grey2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    #orb - stworzenie metody dla wyszukiwania lokalnych cech w obrazach 
    # korzystalem z orb
    orb = cv2.ORB_create()
    
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
    dst += (w, 0) # dodawanie offsetu

    draw_params = dict(matchColor = (0,255,0), # rysowanie matchow na zielono
                       singlePointColor = None,
                       matchesMask = matchesMask, # rysowanie tylko inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None,**draw_params)

    # Rysowanie obwiednię w kolorze czerwonym
    img3 = cv2.polylines(img3, [np.int32(dst)], True, (255,0,0), 3, cv2.LINE_AA)
    #Wyswietlanie zdjecia
    show_img(img3, True)


def main(args):    
    img1 = cv2.imread(args.img1) #zapisywania pierwszego zdjecia
    img2 = cv2.imread(args.img2) #zapisywania drugiego zdjecia
    
    #sprawdzanie czy nie sa pustymi img1 i img2
    #jezeli jakis z elementow jest pusty - zamyka program
    if img1 is None:
        print("img1 is None")
        sys.exit()
    if img2 is None:
        print("img2 is None")
        sys.exit()
        
    #Resize dwoch zdjec dla łatwiejszej pracy uzytkowniku
    # i dla szybszej pracy programu 
    img1 = cv2.resize(img1,(600,600))
    img2 = cv2.resize(img2,(600,600))
    
    #przypisywanie zdjec do metody
    BruteForce(img1,img2)
    
#Metoda do pobrania dwoch zdjec
#-i1 --img1 pierwsze zdjecie
#-i2 --img2 drugie zdjecie
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1','--img1',default=0)
    parser.add_argument('-i2','--img2',default=0)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())

