import json
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Definir las conexiones entre puntos para tu configuración específica de OpenPose
conexiones = [
    (17, 18), (17, 1), (18, 1),
    (1, 5), (1, 2), (5, 6), (2, 3), (3, 4), (6, 7),
    (1, 8), (8, 12), (8, 9), (12, 13), (9, 10), (10, 11),
    (11, 24), (11, 22), (22, 23), (13, 14), (14, 21),
    (19, 20), (14, 19)
]

def visualizar_pose(frame_data, ax, dispersión_umbral=0, min_puntos_pose=0):
    ax.clear()
    ax.invert_yaxis()  # Invertir el eje y para que la parte superior sea 0

    # Calcular la dispersión media de la pose
    dispersión_media = calcular_dispersión_media_por_frame(frame_data)

    # Dibujar las poses y conexiones solo si la dispersión media es mayor que el umbral y tiene al menos 15 puntos
    for i, persona in enumerate(frame_data["people"]):
        keypoints = persona["pose_keypoints_2d"]
        if dispersión_media[i] > dispersión_umbral and contar_puntos_validos(keypoints) >= min_puntos_pose:
            dibujar_pose([(keypoints[j], keypoints[j + 1], keypoints[j + 2]) for j in range(0, len(keypoints), 3)],
                          conexiones, ax)


def contar_puntos_validos(keypoints):
    # Contar puntos con confianza mayor que 0.2
    confianza_coords = keypoints[2::3]
    puntos_validos = sum(1 for confianza in confianza_coords if confianza > 0.2)
    return puntos_validos


def dibujar_pose(pose, conexiones, ax):
    for punto in pose:
        x, y = punto[:2]
        confianza = punto[2]
        
        if confianza > 0.2:
            ax.plot(x, y, 'ro')

    for conexion in conexiones:
        punto_1 = pose[conexion[0]]
        punto_2 = pose[conexion[1]]
        
        if len(punto_1) > 2 and len(punto_2) > 2 and punto_1[2] > 0.2 and punto_2[2] > 0.2:
            x = [punto_1[0], punto_2[0]]
            y = [punto_1[1], punto_2[1]]
            ax.add_line(Line2D(x, y, linewidth=2, color='blue'))

def dibujar_lineas(puntos, conexiones, ax):
    for conexion in conexiones:
        punto_1 = puntos[conexion[0]]
        punto_2 = puntos[conexion[1]]
        
        if len(punto_1) > 2 and len(punto_2) > 2 and punto_1[2] > 0.2 and punto_2[2] > 0.2:
            x = [punto_1[0], punto_2[0]]
            y = [punto_1[1], punto_2[1]]
            ax.add_line(Line2D(x, y, linewidth=2, color='blue'))
            
def calcular_dispersión_media_por_frame(frame_data):
    dispersión_media = []

    for persona in frame_data["people"]:
        keypoints = persona["pose_keypoints_2d"]
        x_coords = keypoints[0::3]
        y_coords = keypoints[1::3]

        # Filtrar puntos con baja confianza
        confianza_coords = keypoints[2::3]
        puntos_filtrados = [(x, y) for x, y, confianza in zip(x_coords, y_coords, confianza_coords) if confianza > 0.2]

        # Calcular dispersión media solo si hay suficientes puntos
        if len(puntos_filtrados) > 1:
            dispersión_media_persona = np.std(puntos_filtrados, axis=0).mean()
            dispersión_media.append(dispersión_media_persona)
        else:
            dispersión_media.append(0.0)  # Asignar 0 si no hay suficientes puntos

    return dispersión_media

ruta_carpeta_frames = r"D:\Users\Alejandro\Downloads\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended (2)\openpose\output_json_folder\AbsoluteMassiveKilldeer"  # Reemplaza con la ruta real de tu carpeta

video_writer = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
fig, ax = plt.subplots()

for archivo_json in sorted(os.listdir(ruta_carpeta_frames)):
    if archivo_json.endswith(".json"):
        ruta_completa = os.path.join(ruta_carpeta_frames, archivo_json)
        with open(ruta_completa, 'r') as f:
            datos = json.load(f)

        # Visualizar el frame solo si la dispersión media es mayor que 45
        visualizar_pose(datos, ax, dispersión_umbral=0)

        # Guardar el frame en el video
        fig.savefig("temp_frame.png")
        img_array = cv2.imread("temp_frame.png")
        video_writer.write(img_array)

video_writer.release()
plt.close()
