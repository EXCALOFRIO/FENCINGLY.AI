import json

# Nombre del archivo JSON de entrada
input_file = "epee_data.json"

# Cargamos los datos desde el archivo JSON utilizando UTF-8
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Creamos dos listas vacías para enlaces de descarga de tocados derecho e izquierdo
foil_right_links = []
foil_left_links = []

# Iteramos sobre los elementos del JSON y almacenamos los enlaces de descarga
for item in data:
    if item["touch"] == "right":
        foil_right_links.append(item["download_url"])
    elif item["touch"] == "left":
        foil_left_links.append(item["download_url"])

# Guardamos los enlaces de descarga en archivos JSON separados
with open("epee_right_links.json", "w") as foil_right_file:
    json.dump(foil_right_links, foil_right_file, indent=4)

with open("eppe_left_links.json", "w") as foil_left_file:
    json.dump(foil_left_links, foil_left_file, indent=4)
