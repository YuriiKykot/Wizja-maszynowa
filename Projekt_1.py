import sys
import cv2
import numpy as np
import argparse

color = None
low = None
high = None
hsv = None
image_mask = False

def mouse_handler(event, x, y, flags, param):
    global color, low, high, hsv, image_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        hue = hsv[y, x, 0]
        saturation = hsv[y, x, 1]
        value = hsv[y, x, 2]
        color = np.array([hue, saturation, value])
        low = np.array([hue - 20, saturation - 50, value - 60])
        high = np.array([hue + 20, saturation + 50, value + 60])

def main(args):
    global color, low, high, hsv, image_mask
    
    window_name = 'Image'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_handler)
    
    video = cv2.VideoCapture(args.input)
    
    kernel = np.ones((4,4), np.uint8)
    
    if video is None:
        sys.exit()
    
    captured, frame = video.read()

    while captured:
        processed = None
        
        captured, frame = video.read()
        frame = cv2.resize(frame,(600,600))
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
           
        if color is not None:
            mask = cv2.inRange(hsv, low, high)
            mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask_closed = cv2.morphologyEx(mask_without_noise, cv2.MORPH_CLOSE, kernel)
            processed = mask_closed
        
        if image_mask and color is not None:
            conts,h=cv2.findContours(processed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for i in range(len(conts)):
                x,y,w,h=cv2.boundingRect(conts[i])
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow(window_name, frame)
        else:
            cv2.imshow(window_name, frame)
            
        key = cv2.waitKey(1)
        if key == ord('q'):
            captured = False
            break
        elif key == ord('m'):
            image_mask = not image_mask
            

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',default=0)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())
