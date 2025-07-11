import os
import shutil

def split_images(source_folder, train_folder, test_folder, num_images_per_set=250):
    # Utwórz folder train, jeśli nie istnieje
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    
    # Utwórz folder test, jeśli nie istnieje
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Pobierz listę plików z folderu źródłowego
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
    
    # Podziel listę plików na zestawy treningowe i testowe
    for i in range(0,  num_images_per_set):
        train_images = image_files[i:i+num_images_per_set]
        test_images = image_files[i+num_images_per_set:i+2*num_images_per_set]
        
        # Przenieś pliki do folderów treningowych i testowych
        for image in train_images:
            source_path = os.path.join(source_folder, image)
            destination_path = os.path.join(train_folder, image)
            shutil.copyfile(source_path, destination_path)
        
        for image in test_images:
            source_path = os.path.join(source_folder, image)
            destination_path = os.path.join(test_folder, image)
            shutil.copyfile(source_path, destination_path)

# Użycie funkcji
split_images("horizontal_bar", "train", "test", num_images_per_set=50)
split_images("vertical_bar", "train", "test", num_images_per_set=50)