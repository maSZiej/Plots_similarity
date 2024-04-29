
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

# Lista przechowująca dane obrazów
images_data = []

# Wczytywanie i przekształcanie obrazów do danych numerycznych
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)  # Wczytanie obrazu w skali szarości
        img_data = img.flatten()  # Spłaszczenie obrazu do jednowymiarowej tablicy
        images_data.append(img_data)

# Konwersja listy na tablicę numpy
images_data = np.array(images_data)

# Tworzenie mapy SOM
som = MiniSom(30, 30, images_data.shape[1], sigma=2, learning_rate=0.8)
som.random_weights_init(images_data)

# Liczba epok treningowych
n_epochs = 1000

som.train(images_data, 100,random_order=True, verbose=True)  
    
