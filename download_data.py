import kagglehub

# Download latest version
path = kagglehub.dataset_download("ferno2/training1600000processednoemoticoncsv")

print("Path to dataset files:", path)