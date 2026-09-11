import json

# Nombre del archivo JSON de entrada
input_file = "json/foil/all_gfy_info copy.json"

# Cargamos los datos desde el archivo JSON utilizando UTF-8
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)
    
# Creamos tres listas vacías para cada arma
foil_data = []
sabre_data = []
epee_data = []

# Iteramos sobre los elementos del JSON y los distribuimos en las listas correspondientes
for item in data:
    if item["weapon"] == "foil":
        foil_data.append(item)
    elif item["weapon"] == "sabre":
        sabre_data.append(item)
    elif item["weapon"] == "epee":
        epee_data.append(item)

# Creamos tres archivos JSON para cada arma
with open("foil_data.json", "w") as foil_file:
    json.dump(foil_data, foil_file, indent=4)

with open("sabre_data.json", "w") as sabre_file:
    json.dump(sabre_data, sabre_file, indent=4)

with open("epee_data.json", "w") as epee_file:
    json.dump(epee_data, epee_file, indent=4)
