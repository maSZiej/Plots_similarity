import cv2
import numpy as np
import pytesseract 
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Wczytanie obrazu
img = cv2.imread(r"C:\Users\Si3ma\chart_analysys\chart_properties_analysis_app\Maciek-Branch\App\Wykresy\wykres_17.png")

# Konwersja obrazu do przestrzeni kolorów HSV (łatwiejsze definiowanie zakresu kolorów)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Zdefiniowanie dolnego i górnego zakresu kolorów dla białego
lower_white = np.array([0, 0, 180])
upper_white = np.array([255, 150, 255])

# Progowanie obrazu w przestrzeni kolorów HSV w celu wyodrębnienia obszarów białych
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Definiowanie zakresu dla koloru czarnego
black_color_low = np.array([0, 0, 0])
black_color_high = np.array([360, 255, 170])
mask_black_color = cv2.inRange(hsv, black_color_low, black_color_high)

# Negacja maski białej i czarnej, aby uzyskać obszary inne niż białe i czarne
mask_not_white = cv2.bitwise_not(mask_white)
mask_not_black = cv2.bitwise_not(mask_black_color)

# Połączenie masek
combined_mask = cv2.bitwise_and(mask_not_white, mask_not_black)

# Znalezienie konturów
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrowanie konturów na podstawie warunków (zaktualizowane warunki filtracji)
filtered_contours = []
min_aspect_ratio = 0.1
min_area = 1
min_height = 0.5

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if aspect_ratio >= min_aspect_ratio and area >= min_area and h >= min_height:
        filtered_contours.append(contour)

# Wycięcie słupków i sortowanie według wysokości
bars = [(cv2.boundingRect(contour), img[y:y+h, x:x+w]) for contour in filtered_contours]
bars.sort(key=lambda b: b[0][3])  # Sortowanie według wysokości (h)

# Utworzenie nowego obrazu o odpowiedniej wysokości
max_height = max([h for (x, y, w, h), _ in bars])+50
total_width = sum([w for (x, y, w, h), _ in bars]) + 15 * (len(bars) - 1)  # Dodanie odstępów między słupkami
sorted_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)
sorted_img.fill(255)  # Wypełnienie tłem na biało

# Funkcja do wykrywania tekstu w obrazie
def extract_text_regions(image):
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d['level'])
    text_boxes = []
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        text_boxes.append((x, y, w, h, d['text'][i]))
    return text_boxes

# Wyodrębnienie obszarów tekstowych
text_boxes = extract_text_regions(img)

# Przypisanie etykiet do słupków na podstawie pozycji
categories = []
for (x, y, w, h), bar in bars:
    assigned_category = "Unknown"
    for (tx, ty, tw, th, text) in text_boxes:
        if tx >= x and tx + tw <= x + w and ty > y + h:
            assigned_category = text
            break
    categories.append(assigned_category)

# Wyświetlenie obrazu z posortowanymi słupkami i odstępami oraz etykietami
current_x =10
bar_spacing = 10  # Odstęp między słupkami
for i, ((x, y, w, h), bar) in enumerate(bars):
    if w > 0:  # Upewnij się, że szerokość słupka jest większa od zera
        bar_resized = cv2.resize(bar, (w, h), interpolation=cv2.INTER_AREA)
        sorted_img[max_height-h:max_height, current_x:current_x+w] = bar_resized
        cv2.putText(sorted_img, categories[i], (current_x, max_height - h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Dodanie tekstu na niebiesko
        current_x += w + bar_spacing

cv2.imshow('Sorted Bars with Spacing and Categories', sorted_img)
cv2.imshow('Combined Mask', combined_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Wypisanie wyników
heights = [b[0][3] for b in bars]
print("Wysokości słupków:", heights)
print("Posortowane wysokości słupków:", sorted(heights))
print("Kategorie słupków:", categories)