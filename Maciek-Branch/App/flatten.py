
#pickle is included in python packages
import pickle
# Import bibliotek
import os
import cv2
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Ścieżka do folderu zawierającego obrazy PNG
folder_path = "Wykresy"



img = cv2.imread("Wykresy\wykres_1.png")

# Konwersja obrazu do przestrzeni kolorów HSV (łatwiejsze definiowanie zakresu kolorów)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Zdefiniowanie dolnego i górnego zakresu kolorów dla białego
lower_white = np.array([0, 0, 180])
upper_white = np.array([255, 150, 255])

# Progowanie obrazu w przestrzeni kolorów HSV w celu wyodrębnienia obszarów białych
mask_white = cv2.inRange(hsv, lower_white, upper_white)


black_color_low=np.array([0,0,0])
black_color_high=np.array([360,255,170])
mask_black_color= cv2.inRange(hsv,black_color_low, black_color_high)


mask_not_black=cv2.bitwise_not(mask_black_color)
# Negacja maski białej, aby uzyskać obszary inne niż białe
mask_not_white = cv2.bitwise_not(mask_white)

# Progowanie obrazu w skali szarości w celu wyodrębnienia obszarów czarnych
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask_black = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

# Połączenie maski czarnej i negowanej maski białej, aby uzyskać obszary inne niż czarne i białe
combined_mask = cv2.bitwise_and(mask_not_white, mask_not_black)
cv2.imshow('img',img)
cv2.imshow('combined',combined_mask)
cv2.imshow('white',mask_not_white)
cv2.imshow('black',mask_not_black)

cv2.waitKey(0)


