import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import os
import cv2
import pytesseract
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

from sklearn_som.som import SOM  # Zakładając, że sklearn_som jest zainstalowany
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_features(image_path):
    # Wczytanie obrazu
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    # Konwersja obrazu do skali szarości
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Obliczenie transformaty Fouriera 2D
    fft_image = fft2(gray_image)
    
    # Przesunięcie zerowe częstotliwości do środka
    fft_shifted = fftshift(fft_image)
    
    # Znormalizowanie widma do wartości bezwzględnych
    magnitude_spectrum = np.abs(fft_shifted)
    
    # Zwrot znormalizowanego widma
    return magnitude_spectrum

folder_path = "train"

# Ekstrakcja cech z obrazów
features = []
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(folder_path, filename)
        features.append(extract_features(image_path))

features = np.array(features)

som_size1 = 10
som_size2 = 10
print(features)
features_normalized = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
# Definiowanie i trenowanie SOM
som = SOM(m=som_size1, n=som_size2, dim=224, sigma=0.5, lr=0.5)
som.fit(features_normalized, epochs=3500)

# Pobranie klastrów z SOM
bmus = som.predict(features_normalized)

unique_bmus = np.unique(bmus)
cluster_centroids = np.array([features[bmus == i].mean(axis=0) for i in unique_bmus])

# Apply hierarchical clustering to the cluster centroids
linkage_matrix = linkage(cluster_centroids, method='ward')

# Ocena jakości klasteryzacji
if len(np.unique(bmus)) > 1:
    silhouette_avg = silhouette_score(features_normalized, bmus)
    davies_bouldin_avg = davies_bouldin_score(features_normalized, bmus)
    calinski_harabasz_avg = calinski_harabasz_score(features_normalized, bmus)

    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_avg}")
else:
    print("Number of unique clusters is less than 2. Cannot compute cluster quality metrics.")

# Plot dendrogram
plt.title("Hierarchical Clustering Dendrogram for SOM Clusters")
#plot_dendrogram(linkage_matrix, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# Additional plot for understanding which BMUs are chosen
plt.figure(figsize=(3, 3))
plt.title('BMUs Distribution')
plt.hist(bmus, bins=np.arange(0, som_size1 * som_size2 + 1) - 0.5, edgecolor='black')
plt.xlabel('BMU Index')
plt.ylabel('Frequency')
plt.show()

# Visualization of SOM map
plt.figure(figsize=(10, 10))
for i, bmu in enumerate(bmus):
    x, y = divmod(bmu, som_size2)
    plt.text(x + 0.5, y + 0.5, str(i), color=plt.cm.tab10(i / float(som_size1 * som_size2)),
             fontdict={'weight': 'bold', 'size': 9})

plt.title('SOM Map')
plt.xlim([0, som_size1])
plt.ylim([0, som_size2])
plt.grid()
plt.show()

# Function to predict images in a folder
def predict_folder_images(folder_path):
    predicted_clusters = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            predicted_cluster = predict_image(image_path)
            predicted_clusters.append(predicted_cluster)
    return predicted_clusters

# Function to predict image
def predict_image(image_path):
    prob = extract_features(image_path)
    features = np.array([prob])
    predicted_cluster = som.predict(features)
    return predicted_cluster

# Example usage
folder_path = "test"
predicted_clusters = predict_folder_images(folder_path)
print(predicted_clusters)
