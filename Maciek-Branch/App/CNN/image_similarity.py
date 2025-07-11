import torch
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.cluster.hierarchy import dendrogram, linkage
from kmeans_pytorch import kmeans
from PIL import Image
from sklearn.decomposition import PCA
from sklearn_som.som import SOM  # Zakładając, że sklearn_som jest zainstalowany
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

#from som import SOM

from sklearn.preprocessing import MinMaxScaler

class Img2Vec:
    """
    Class for embedding dataset of image files into vectors using Pytorch
    standard neural networks.

    Parameters:
    -----------
    model_name: str object specifying neural network architecture to utilise.
        Must align to the naming convention specified on Pytorch documentation:
        https://pytorch.org/vision/main/models.html#classification
        For supported model architectures see self.embed_dict below
    weights: str object specifying the pretrained weights to load into model.
        Only weights supported by Pytorch torchvision library can be accessed.
        Current functionality reverts to DEFAULT weights if no specified.

    See also:
    -----------
    Img2Vec.embed_dataset(): embed passed images as feature vectors
    Img2Vec.save_dataset(): save embedded dataset to file for future loading
    Img2Vec.load_dataset(): load previously embedded dataset of feature vectors
    Img2Vec.similar_image(): pass target image and return most similar image(s)
    Img2Vec.cluster_dataset(): group embedded images into specified n clusters

    Example:
    -----------

    ImgSim = imgsim.Img2Vec('resnet50', weights='DEFAULT')
    ImgSim.embed_dataset('[EXAMPLE PATH TO DIRECTORY OF IMAGES]')

    ImgSim.save_dataset('[OUTPUT PATH FOR SAVING EMBEDDEDINGS]')

    ImgSim.similar_images('[EXAMPLE PATH TO TARGET IMAGE]')

    ImgSim.cluster_dataset(nclusters=6, display=True)
    """

    def __init__(self, model_name, weights="DEFAULT"):
        # dictionary defining the supported NN architectures
        self.embed_dict = {
            "resnet50": self.obtain_children,
            "vgg19": self.obtain_classifier,
            "efficientnet_b0": self.obtain_classifier,
        }

        # assign class attributes
        self.architecture = self.validate_model(model_name)
        self.weights = weights
        self.transform = self.assign_transform(weights)
        self.device = self.set_device()
        self.model = self.initiate_model()
        self.embed = self.assign_layer()
        self.dataset = {}
        self.image_clusters = {}
        self.cluster_centers = {}

    def validate_model(self, model_name):
        if model_name not in self.embed_dict.keys():
            raise ValueError(f"The model {model_name} is not supported")
        else:
            return model_name

    def assign_transform(self, weights):
        weights_dict = {
            "resnet50": models.ResNet50_Weights,
            "vgg19": models.VGG19_Weights,
            "efficientnet_b0": models.EfficientNet_B0_Weights,
        }

        # try load preprocess from torchvision else assign default
        try:
            w = weights_dict[self.architecture]
            weights = getattr(w, weights)
            preprocess = weights.transforms()
        except Exception:
            preprocess = transforms.Compose(
                [
                    transforms.Resize(900),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        return preprocess

    def set_device(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        return device

    def initiate_model(self):
        m = getattr(
            models, self.architecture
        )  # equ to assigning m as models.resnet50()
        model = m(weights=self.weights)  # equ to models.resnet50(weights=...)
        model.to(self.device)

        return model.eval()

    def assign_layer(self):
        model_embed = self.embed_dict[self.architecture]()

        return model_embed

    def obtain_children(self):
        model_embed = nn.Sequential(*list(self.model.children())[:-1])

        return model_embed

    def obtain_classifier(self):
        self.model.classifier = self.model.classifier[:-1]

        return self.model

    def directory_to_list(self, dir):
        ext = (".png", ".jpg", ".jpeg")

        d = os.listdir(dir)
        source_list = [os.path.join(dir, f) for f in d if os.path.splitext(f)[1] in ext]

        return source_list

    def validate_source(self, source):
        # convert source format into standard list of file paths
        if isinstance(source, list):
            source_list = [f for f in source if os.path.isfile(f)]
        elif os.path.isdir(source):
            source_list = self.directory_to_list(source)
        elif os.path.isfile(source):
            source_list = [source]
        else:
            raise ValueError('"source" expected as file, list or directory.')

        return source_list

    def embed_image(self, img):
        # load and preprocess image
        img = Image.open(img)
        img_trans = self.transform(img)

        # store computational graph on GPU if available
        if self.device == "cuda:0":
            img_trans = img_trans.cuda()

        img_trans = img_trans.unsqueeze(0)

        return self.embed(img_trans)

    def embed_dataset(self, source):
        # convert source to appropriate format
        self.files = self.validate_source(source)

        for file in self.files:
            vector = self.embed_image(file)
            self.dataset[str(file)] = vector

        return

    def similar_images(self, target_file, n=None):
        """
        Function for comparing target image to embedded image dataset

        Parameters:
        -----------
        target_file: str specifying the path of target image to compare
            with the saved feature embedding dataset
        n: int specifying the top n most similar images to return
        """

        target_vector = self.embed_image(target_file)

        # initiate computation of consine similarity
        cosine = nn.CosineSimilarity(dim=1)

        # iteratively store similarity of stored images to target image
        sim_dict = {}
        for k, v in self.dataset.items():
            sim = cosine(v, target_vector)[0].item()
            sim_dict[k] = sim

        # sort based on decreasing similarity
        items = sim_dict.items()
        sim_dict = {k: v for k, v in sorted(items, key=lambda i: i[1], reverse=True)}

        # cut to defined top n similar images
        if n is not None:
            sim_dict = dict(list(sim_dict.items())[: int(n)])

        self.output_images(sim_dict, target_file)

        return sim_dict

    def output_images(self, similar, target):
        self.display_img(target, "original")

        for k, v in similar.items():
            self.display_img(k, "similarity:" + str(v))

        return

    def display_img(self, path, title):
        plt.imshow(Image.open(path))
        plt.axis("off")
        plt.title(title)
        plt.show()

        return

    def save_dataset(self, path):
        """
        Function to save a previously embedded image dataset to file

        Parameters:
        -----------
        path: str specifying the output folder to save the tensors to
        """

        # convert embeddings to dictionary
        data = {"model": self.architecture, "embeddings": self.dataset}

        torch.save(
            data, os.path.join(path, "tensors.pt")
        )  # need to update functionality for naming convention

        return

    def load_dataset(self, source):
        """
        Function to save a previously embedded image dataset to file

        Parameters:
        -----------
        source: str specifying tensor.pt file to load previous embeddings
        """

        data = torch.load(source)

        # assess that embedding nn matches currently initiated nn
        if data["model"] == self.architecture:
            self.dataset = data["embeddings"]
        else:
            raise AttributeError(
                f'NN architecture "{self.architecture}" does not match the '
                + f'"{data["model"]}" model used to generate saved embeddings.'
                + " Re-initiate Img2Vec with correct architecture and reload."
            )

        return

    def plot_list(self, img_list, cluster_num):
        fig, axes = plt.subplots(math.ceil(len(img_list) / 2), 2)
        fig.suptitle(f"Cluster: {str(cluster_num)}")
        [ax.axis("off") for ax in axes.ravel()]

        for img, ax in zip(img_list, axes.ravel()):
            ax.imshow(Image.open(img))

        fig.tight_layout()

        return
    '''
    def display_clusters(self):
        for num in self.cluster_centers.keys():
            # print(f'Displaying cluster: {str(cluster_num)}')

            img_list = [k for k, v in self.image_clusters.items() if v == num]
            self.plot_list(img_list, num)

        return
    '''
    def display_clusters(self):
        for cluster_num, cluster_center in self.cluster_centers.items():
            print(f"Cluster {cluster_num}:")
            img_list = [k for k, v in self.image_clusters.items() if v == cluster_num]
            print(f"Number of images in cluster: {len(img_list)}")

            # Display first 5 images from the cluster
            if len(img_list) > 5:
                img_list = img_list[:5]

            self.plot_list(img_list, cluster_num)

        return

    def cluster_dataset(self, nclusters, dist="euclidean", display=False):
        vecs = torch.stack(list(self.dataset.values())).squeeze()
        imgs = list(self.dataset.keys())
        np.random.seed(100)

        cluster_ids_x, cluster_centers = kmeans(
            X=vecs, num_clusters=nclusters, distance=dist, device=self.device
        )

        # assign clusters to images
        self.image_clusters = dict(zip(imgs, cluster_ids_x.tolist()))

        # store cluster centres
        cluster_num = list(range(0, len(cluster_centers)))
        self.cluster_centers = dict(zip(cluster_num, cluster_centers.tolist()))

        if display:
            self.display_clusters()

        return


    def train_som(self, n_rows, n_columns, epochs,sigma,lr):

        # Przygotowanie danych do SOM
        embeddings = list(self.dataset.values())
        embeddings = torch.stack(embeddings).squeeze()  # Konwersja do tensorów PyTorch
        print("Embeddings :", embeddings)
        
        scaler = MinMaxScaler()
        embeddings_normalized = scaler.fit_transform(embeddings.detach().numpy())
        print("normalized: ",embeddings_normalized)
        # Normalizacja danych na tensorach PyTorch
        print(type(embeddings_normalized))
        # Trenowanie SOM
        self.som = SOM(m=n_rows, n=n_columns, dim=embeddings_normalized.shape[1], sigma=sigma, lr=lr)
        self.som.fit(embeddings_normalized,epochs=epochs)
        bmus = self.som.predict(embeddings_normalized)
            # Mapowanie danych na klaster BMU (Best Matching Unit)


        # Obliczenie Silhouette Score
        silhouette_avg = silhouette_score(embeddings_normalized, bmus)
        print(f'Silhouette Score: {silhouette_avg}')

        # Obliczenie Calinski-Harabasz Index
        calinski_harabasz_avg = calinski_harabasz_score(embeddings_normalized, bmus)
        print(f'Calinski-Harabasz Index: {calinski_harabasz_avg}')

        # Obliczenie Davies-Bouldin Index
        davies_bouldin_avg = davies_bouldin_score(embeddings_normalized, bmus)
        print(f'Davies-Bouldin Index: {davies_bouldin_avg}')

        # Redukcja wymiarów za pomocą PCA
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings_normalized)

        # Wizualizacja klastrów
        plt.figure(figsize=(10, 8))
        unique_bmus = np.unique(bmus)
        for i, bmu in enumerate(unique_bmus):
            cluster_points = embeddings_pca[bmus == bmu]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {bmu}')

        plt.title('Clusters Visualization with PCA')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.show()

        self.plot_som_clusters(embeddings_normalized=embeddings_normalized,bmus=bmus,n_columns=n_columns,n_rows=n_rows)
        weights = self.som.weights

        weights_reshaped = weights.reshape((n_rows, n_columns, -1))
        print("Weights reshaped:", weights_reshaped.shape)

        # Create a heatmap of the weights
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(np.linalg.norm(weights_reshaped, axis=-1), cmap='viridis')



        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('SOM Visualization (Heatmap)')


        plt.show()
        '''
        mapped = self.som.map_vects(torch.Tensor(embeddings_normalized))
        print("Mapped: ",type(mapped))
        # Tworzymy słownik, gdzie kluczem jest indeks BMU, a wartością jest lista wektorów wejściowych przypisanych do tego klastra
        clusters = {}
        for i, m in enumerate(mapped):
            index = tuple(m.numpy())  # Konwertujemy indeks BMU na krotkę, aby można go było użyć jako klucza w słowniku
            if index in clusters:
                clusters[index].append(embeddings_normalized[i])
            else:
                clusters[index] = [embeddings_normalized[i]]

        print("Clusters:", clusters)
        visualize_clusters(clusters)
                '''
        return  dict(zip(list(self.dataset.keys()), bmus))


    import matplotlib.pyplot as plt



    def predict_som(self):
        if not hasattr(self, 'som'):
            raise AttributeError("SOM model has not been trained yet. Please train SOM first.")
        
        # Przewidywanie klastrów na danych z SOM
        embeddings = list(self.dataset.values())
       
        embeddings = torch.stack(embeddings).squeeze()  # Konwersja do tensorów PyTorch
         # Normalizacja danych na tensorach PyTorch
        '''

        embeddings_min = embeddings.min(dim=0, keepdim=True)[0]
        embeddings_max = embeddings.max(dim=0, keepdim=True)[0]
        embeddings_normalized = (embeddings - embeddings_min) / (embeddings_max - embeddings_min)
        embeddings_normalized = embeddings_normalized.detach().numpy()
        '''
                # Normalizacja danych

        scaler = MinMaxScaler()
        embeddings_normalized = scaler.fit_transform(embeddings.numpy())
        bmus = self.som.predict(embeddings_normalized)
        self.plot_som_clusters(embeddings_normalized=embeddings_normalized,bmus=bmus)
        
        # Zwracanie przypisanych klastrów do każdego obrazu
        return dict(zip(list(self.dataset.keys()), bmus))
    


    def plot_som_clusters(self,bmus,embeddings_normalized,n_rows, n_columns,):
        """
        Plot the SOM clusters and visualize the BMUs.
        """
        unique_bmus = np.unique(bmus)
        cluster_centroids = np.array([embeddings_normalized[bmus == i].mean(axis=0) for i in unique_bmus])

        # Apply hierarchical clustering to the cluster centroids
        linkage_matrix = linkage(cluster_centroids, method='ward')

        # Plot dendrogram
        plt.title("Hierarchical Clustering Dendrogram for SOM Clusters")
        dendrogram(linkage_matrix, truncate_mode="level", p=10)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

        # Additional plot for understanding which BMUs are chosen
        plt.figure(figsize=(10, 8))
        plt.title('BMUs Distribution')
        plt.hist(bmus, bins=np.arange(0, n_columns* n_rows+ 1) - 0.5, edgecolor='black')
        plt.xlabel('BMU Index')
        plt.ylabel('Frequency')
        plt.show()


