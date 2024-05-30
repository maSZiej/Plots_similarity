import os
import cv2
import numpy as np
from minisom import MiniSom
import pickle

# Ścieżka do folderu zawierającego obrazy PNG
folder_path = "Wykresy"

# Lista do przechowywania wysokości słupków ze wszystkich obrazów
all_heights = []

# Próg minimalnej wysokości słupka
min_height = 0.5

# Przetwarzanie każdego obrazu w folderze
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        img = cv2.imread(os.path.join(folder_path, filename))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


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
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h >= min_height:
                all_heights.append(h)

# Konwersja do tablicy numpy
all_heights = np.array(all_heights)

# Normalizacja wysokości słupków
if all_heights.max() != all_heights.min():
    all_heights = (all_heights - all_heights.min()) / (all_heights.max() - all_heights.min())
else:
    all_heights = all_heights / all_heights.max()

# Trening SOM
som = MiniSom(10, len(all_heights), 1, sigma=2, learning_rate=2)
som.random_weights_init(all_heights.reshape(-1, 1))
som.train_random(all_heights.reshape(-1, 1), 10000)

# Zapisanie wytrenowanego modelu SOM
with open("trained_som.pkl", "wb") as f:
    pickle.dump(som, f)

print("Trening SOM zakończony i zapisany do pliku trained_som.pkl")