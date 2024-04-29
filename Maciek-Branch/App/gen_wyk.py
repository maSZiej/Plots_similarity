import os
import matplotlib.pyplot as plt
import numpy as np

# Tworzenie folderu, jeśli nie istnieje
folder_name = "Wykresy"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Pętla generująca i zapisująca wykresy słupkowe
for i in range(1, 21):
    # Generowanie losowych danych dla wykresu słupkowego
    categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
    values = np.random.randint(1, 100, size=len(categories))
    
    # Tworzenie wykresu słupkowego
    plt.bar(categories, values)
    plt.title(f'Wykres {i}')
    plt.xlabel('Kategoria')
    plt.ylabel('Wartość')
    
    # Zapisywanie wykresu do pliku w wybranym folderze
    filename = os.path.join(folder_name, f'wykres_{i}.png')
    plt.savefig(filename)
    plt.close()  # Zamykanie bieżącego wykresu, aby uniknąć nakładania się na kolejne wykresy
    
    print(f'Zapisano wykres do pliku: {filename}')

print("Wszystkie wykresy zostały wygenerowane i zapisane.")
