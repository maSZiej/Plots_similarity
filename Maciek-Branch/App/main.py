from fastapi import FastAPI, UploadFile, File
import numpy as np
from fastapi.responses import StreamingResponse
import pickle
import os
import cv2
from minisom import MiniSom
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io


import io
app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/')
async def upload_image(image: UploadFile = File(...)):
    with open("trained_som.pkl", "rb") as f:
        som = pickle.load(f)

    # Tutaj możesz wykonać operacje na przesłanym obrazie
    contents = await image.read()

    # Konwersja danych obrazu na obiekt PIL Image
    pil_image = Image.open(io.BytesIO(contents))

    # Konwersja obiektu PIL Image do tablicy numpy
    np_image = np.array(pil_image)
    np_image =cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    # Konwersja obrazu do przestrzeni kolorów HSV (łatwiejsze definiowanie zakresu kolorów)
    hsv = cv2.cvtColor(np_image, cv2.COLOR_BGR2HSV)

    # Zdefiniowanie dolnego i górnego zakresu kolorów dla białego
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 150, 255])

    # Progowanie obrazu w przestrzeni kolorów HSV w celu wyodrębnienia obszarów białych
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    black_color_low = np.array([0, 0, 0])
    black_color_high = np.array([360, 255, 170])
    mask_black_color = cv2.inRange(hsv, black_color_low, black_color_high)

    mask_not_black = cv2.bitwise_not(mask_black_color)
    # Negacja maski białej, aby uzyskać obszary inne niż białe
    mask_not_white = cv2.bitwise_not(mask_white)

    # Progowanie obrazu w skali szarości w celu wyodrębnienia obszarów czarnych
    gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    _, mask_black = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # Połączenie maski czarnej i negowanej maski białej, aby uzyskać obszary inne niż czarne i białe
    combined_mask = cv2.bitwise_and(mask_not_white, mask_not_black)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Minimalny stosunek długości do szerokości
    min_aspect_ratio = 0.1

    # Minimalne pole powierzchni
    min_area = 1

    # Minimalny stosunek obwodu do pola powierzchni
    min_contour_ratio = 0

    # Filtrowanie konturów
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if aspect_ratio >= min_aspect_ratio and area >= min_area and perimeter > 0:
            contour_ratio = perimeter / area
            if contour_ratio >= min_contour_ratio:
                filtered_contours.append(contour)

    contours=filtered_contours




    heights = []
    min_height = 0.5
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h >= min_height:
            heights.append(h)
    heights = np.array(heights)
    heights = (heights - heights.min()) / (heights.max() - heights.min())

        # Uzyskanie wyników klasteryzacji
    winning_nodes = np.array([som.winner(h.reshape(1, 1)) for h in heights])

    # Sprawdzenie kolejności słupków
    order = np.argsort(winning_nodes[:, 1])
    sorted_heights = heights[order]

    # Klasyfikacja typów słupków
    trends = []
    for i in range(1, len(heights)):
        if heights[i] > heights[i - 1]:
            trends.append('rosnący')
        elif heights[i] < heights[i - 1]:
            trends.append('malejący')
        else:
            trends.append('stały')

    # Wypisanie wyników
    print("Wysokości słupków: \n", heights)
    print("Posortowane wysokości słupków:", sorted_heights)
    print("Trendy słupków:", trends)


    # Wizualizacja wyników
    colors = {'rosnący': (0, 255, 0), 'malejący': (0, 0, 255), 'stały': (255, 255, 0)}
    for i, contour in enumerate(contours):
        if i < len(trends):
            color = colors[trends[i]]
            # Wyszukanie prostokąta opisującego dla konturu
            x, y, w, h = cv2.boundingRect(contour)
            # Rysowanie prostokąta opisującego
            cv2.rectangle(np_image, (x, y), (x + w, y + h), color, 2)



    # Wyświetlenie obrazu z wyfiltrowanymi konturami

    # Obliczenie średniej wysokości wszystkich słupków
    mean_height = np.mean(heights)

    # Oznaczenie typów słupków na podstawie ich wysokości w porównaniu do średniej
    trends = []
    for height in heights:
        if height > mean_height:
            trends.append("rosnący")
        elif height < mean_height:
            trends.append("malejący")
        else:
            trends.append("stały")

    # Wyświetlenie wyników
    for i, trend in enumerate(trends):
        print(f"Słupek {i+1} jest typu {trend}")


        # Dodanie tekstowych etykiet nad każdym słupkiem
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(np_image, str(round(heights[i], 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)




    processed_image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB))

    # Zapisz obraz PIL do obiektu io.BytesIO jako PNG
    output = io.BytesIO()
    processed_image.save(output, format="PNG")
    output.seek(0)


    # Zwróć obraz PNG jako odpowiedź strumieniową
    return StreamingResponse(output, media_type="image/png")