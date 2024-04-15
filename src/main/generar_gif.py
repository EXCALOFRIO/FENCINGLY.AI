import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import imageio
import time

conexiones = [
    (17, 18), (17, 1), (18, 1),
    (1, 5), (1, 2), (5, 6), (2, 3), (3, 4), (6, 7),
    (1, 8), (8, 12), (8, 9), (12, 13), (9, 10), (10, 11),
    (11, 24), (11, 22), (22, 23), (13, 14), (14, 21),
    (19, 20), (14, 19)
]

def dibujar_pose(pose, conexiones, ax):
    for i, punto in enumerate(pose):
        if len(punto) < 3:
            continue  # Saltar si el punto no tiene suficientes elementos
        x, y = punto[:2]
        confianza = punto[2]

        if confianza > 0:
            ax.plot(x, y, 'ro', markersize=8)

    for conexion in conexiones:
        if any(index >= len(pose) for index in conexion):
            continue  # Saltar si la conexión tiene un índice inválido

        punto_1 = pose[conexion[0]]
        punto_2 = pose[conexion[1]]

        if len(punto_1) < 3 or len(punto_2) < 3:
            continue  # Saltar si los puntos no tienen suficientes elementos

        if punto_1[2] > 0 and punto_2[2] > 0:
            x = [punto_1[0], punto_2[0]]
            y = [punto_1[1], punto_2[1]]
            ax.add_line(Line2D(x, y, linewidth=3, color='blue'))

def visualizar_tensor_poses(tensor_poses, ax):
    ax.clear()
    ax.set_xlim(0, 1)  # Establecer límites en el eje x
    ax.set_ylim(0, 1)  # Establecer límites en el eje y
    ax.invert_yaxis()  # Invertir el eje y para que la parte superior sea 0

    for persona_poses in tensor_poses:
        for i, persona in enumerate(persona_poses):
            keypoints = persona[:75]  # Tomar solo los primeros 75 keypoints
            dibujar_pose([(keypoints[j], keypoints[j + 1], keypoints[j + 2]) for j in range(0, len(keypoints), 3)],
                          conexiones, ax)


def gif(poses):
    fig, ax = plt.subplots(figsize=(10, 8))
    frames = []
    for frame_poses in poses:
        ax.clear()
        visualizar_tensor_poses([frame_poses], ax)
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(rgba)

    timestamp = str(int(time.time()))  # Obtiene la marca de tiempo actual en segundos
    gif_path = f'poses_animation_{timestamp}.gif'  # Nombre de archivo único con marca de tiempo
    imageio.mimsave(gif_path, frames, fps=5)
    plt.close(fig)  # Cerrar la figura para liberar memoria

    return gif_path
