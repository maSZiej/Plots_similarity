import image_similarity as imgsim

ImgSim = imgsim.Img2Vec('resnet50', weights='DEFAULT')
# observe several of the class attributes post-initialisation
print(ImgSim.architecture)
print(ImgSim.weights)
print(ImgSim.transform)
print(ImgSim.device)
print(ImgSim.model)
print(ImgSim.embed)
ImgSim.embed_dataset(r"C:\Users\Si3ma\chart_analysys\chart_properties_analysis_app\Maciek-Branch\App\CNN\ticker")
ImgSim.dataset


ImgSim.similar_images(r"C:\Users\Si3ma\chart_analysys\chart_properties_analysis_app\Maciek-Branch\App\CNN\test\Chart_20230515221736.jpg", n=5)
ImgSim.cluster_dataset(nclusters=2, display=True)