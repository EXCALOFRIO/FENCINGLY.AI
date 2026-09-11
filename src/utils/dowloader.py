import json
import requests
import os

# Nombre del archivo JSON con los enlaces de video
json_file = "json/foil/foil_right_links copy.json"

# Cargamos los enlaces desde el archivo JSON
with open(json_file, "r") as file:
    video_links = json.load(file)

# Directorio de destino para guardar los videos descargados
download_directory = "videos/"

# Verificamos si el directorio de descarga existe, y si no, lo creamos
if not os.path.exists(download_directory):
    os.makedirs(download_directory)

# Iterar a través de los enlaces y descargar los videos
for link in video_links:
    # Extraer el nombre del archivo desde el enlace
    file_name = link.split("/")[-1]
    file_path = os.path.join(download_directory, file_name)

    # Descargar el video y guardar en el directorio de destino
    response = requests.get(link, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Descargado: {file_name}")
    else:
        print(f"Fallo al descargar: {file_name}")
