import kagglehub

# Download latest version
path = kagglehub.dataset_download("gauravdhamane/gwa-bitbrains")

print("Path to dataset files:", path)