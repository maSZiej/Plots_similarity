import pickle
import os
import cv2
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Ścieżka do folderu zawierającego obrazy PNG
folder_path = "Wykresy"

# Wczytanie obrazu
img = cv2.imread(os.path.join(folder_path, "wykres_1.png"))

# Konwersja obrazu do przestrzeni kolorów HSV (łatwiejsze definiowanie zakresu kolorów)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Zdefiniowanie dolnego i górnego zakresu kolorów dla białego
lower_white = np.array([0, 0, 180])
upper_white = np.array([255, 150, 255])

# Progowanie obrazu w przestrzeni kolorów HSV w celu wyodrębnienia obszarów białych
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Znalezienie konturów białych słupków
contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Ekstrakcja wysokości słupków
heights = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    heights.append(h)

# Normalizacja wysokości słupków
heights = np.array(heights)
heights = (heights - heights.min()) / (heights.max() - heights.min())

# Klasteryzacja przy użyciu MiniSom
som = MiniSom(1, len(heights), 1, sigma=0.3, learning_rate=0.5)
som.random_weights_init(heights.reshape(-1, 1))
som.train_random(heights.reshape(-1, 1), 100)

# Uzyskanie wyników klasteryzacji
winning_nodes = np.array([som.winner(h.reshape(1, 1)) for h in heights])

# Sprawdzenie, czy słupki są ułożone w sposób rosnący czy malejący
order = np.argsort(winning_nodes[:, 1])
sorted_heights = heights[order]

# Wypisanie wyników
print("Wysokości słupków:", heights)
print("Posortowane wysokości słupków:", sorted_heights)

if np.all(sorted_heights[:-1] <= sorted_heights[1:]):
    print("Słupki są ułożone w sposób rosnący.")
elif np.all(sorted_heights[:-1] >= sorted_heights[1:]):
    print("Słupki są ułożone w sposób malejący.")
else:
    print("Słupki są ułożone w sposób mieszany.")

cv2.imshow('img',mask_white)

cv2.waitKey(0)
