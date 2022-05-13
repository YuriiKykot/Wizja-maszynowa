import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DIRECTORY = 'Coins'
coins_images = []
min_x = 0
min_y = 0
max_x = 0
max_y = 0

for entry in os.scandir(DIRECTORY):
    if entry.path.endswith('.jpg') and entry.is_file():
        try:
            img_data = cv2.imread(entry.path)
            coins_images.append(img_data)
        except Exception:
            pass

def find_coins(image, switch):
    # zmiana na skalę szarości
    conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pozbycie się zewnętrznych pikseli przez rozmycie
    blur = cv2.GaussianBlur(conv_img, (5, 5), 2)
    # obliczenie zakresu do stworzenia krawędzi na małych obszarach obrazu
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    # operacje morfologiczne
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    # wykrycie krawędzi
    outline = cv2.Canny(k_open, 100, 100, L2gradient=True)
    # wykrycie okręgów przy użyciu transformaty Hougha
    coins = cv2.HoughCircles(outline, cv2.HOUGH_GRADIENT, 1.5, minDist=30, minRadius=0, maxRadius=50)
    
    if coins is not None:
        coins = np.round(coins[0,:]).astype("int")
        for (x, y, r) in coins:
            cv2.circle(image, (x, y), r, (0,0,0), 6)
    else:
        print("No circles found")
        
    # posortowanie okręgów po promieniu
    coins = coins[np.argsort(coins[:,2])]
    print(len(coins))
    
    suma = 0
    for coin in coins:
        if switch == 1:
            if(coin[2] >= 35):
                cv2.circle(image, (coin[0], coin[1]), coin[2], (0,0,255), 6)
                suma += 5
            else:
                suma += 0.05
        else:
            if(coin[2] >= 32):
                cv2.circle(image, (coin[0], coin[1]), coin[2], (0,0,255), 6)
                suma += 5
            else:
                suma += 0.05
            
    print("Suma {}".format(suma))
            
    #cv2.circle(image, (coins[-1][0], coins[-1][1]), coins[-1][2], (0,0,255), 6)
    #cv2.circle(image, (coins[-2][0], coins[-2][1]), coins[-2][2], (0,0,255), 6)
        
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()  

def find_tray(image):
    global min_x, min_y, max_x, max_y
    # zmiana na skalę szarości
    conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pozbycie się zewnętrznych pikseli przez rozmycie
    blur = cv2.GaussianBlur(conv_img, (5, 5), 2)
    # obliczenie zakresu do stworzenia krawędzi na małych obszarach obrazu
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    # operacje morfologiczne
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    # wykrycie krawędzi
    outline = cv2.Canny(k_open, 100, 100, L2gradient=True)
    # wykrycie lini przy użyciu transformaty Hougha
    lines = cv2.HoughLinesP(outline, 1, np.pi/180, 100, minLineLength=50, maxLineGap=20)
    
    min_x, min_y, max_x, max_y = lines[0][0]
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 < min_x:
            min_x = x1
        if y1 < min_x:
            min_y = y1
        if x2 > max_x:
            max_x = x2
        if y2 > max_y:
            max_y = y2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(conv_img, (x1, y1), (x2, y2), (0,0,255), 6)
        
    #conv_img = cv2.rectangle(conv_img, (min_x, min_y), (max_x,max_y), (0,255,255), 2)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(cv2.cvtColor(conv_img, cv2.COLOR_GRAY2RGB))
    plt.show()  

i = 1
for coins in coins_images:
    find_tray(coins)

    img_1 = coins
    img_2 = coins

    img_inside = img_1[min_y:max_y, min_x:max_x]
    print("Monet na tacy na zdjeciu {}:".format(i))
    find_coins(img_inside, 0)

    img_out = cv2.rectangle(img_2, (min_x, min_y), (max_x,max_y), (0,255,255), -1)
    print("Monet poza na zdjeciu {}:".format(i))
    find_coins(img_out, 1)
    i+=1



