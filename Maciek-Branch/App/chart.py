import matplotlib.pyplot as plt
import numpy as np
import os

# Ścieżka do zapisu wykresów
output_folder = "GeneratedCharts"
os.makedirs(output_folder, exist_ok=True)

def generate_increasing_bar_chart(filename):
    # Tworzenie danych
    num_bars = 10
    heights = np.sort(np.random.randint(1, 100, num_bars))

    # Tworzenie wykresu słupkowego
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_bars), heights)
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Increasing Bar Chart')

    # Zapisywanie wykresu jako obrazu PNG
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path, format='png')
    plt.close()

# Generowanie wykresu słupkowego
generate_increasing_bar_chart('increasing_bar_chart.png')

print(f'Wykres słupkowy został zapisany w folderze {output_folder}')
