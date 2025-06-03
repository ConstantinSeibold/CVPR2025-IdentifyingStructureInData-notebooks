import wget

url = "https://archive.org/download/planetearth2006_202301/%2802%29%20Mountains.mp4"
filename = wget.download(url)

print(f"\nDownloaded file saved as: {filename}")
