import imageio
import numpy as np

def unir_gifs(gif_original_path, gif_transformado_path, output_path):
    # Obtener un lector para los GIFs originales
    reader_original = imageio.get_reader(gif_original_path)
    reader_transformado = imageio.get_reader(gif_transformado_path)
    
    # Obtener el número mínimo de frames entre ambos GIFs
    min_frames = min(len(reader_original), len(reader_transformado))
    
    # Crear un escritor para el nuevo GIF
    writer = imageio.get_writer(output_path, duration=0.1)
    
    # Combinar los GIFs frame por frame
    for i in range(min_frames):
        frame_original = reader_original.get_data(i)
        frame_transformado = reader_transformado.get_data(i)
        
        # Asegurarse de que ambos frames tengan la misma altura
        min_height = min(frame_original.shape[0], frame_transformado.shape[0])
        frame_original = frame_original[:min_height, :]
        frame_transformado = frame_transformado[:min_height, :]
        
        # Combinar los frames horizontalmente
        combined_frame = np.concatenate((frame_original, frame_transformado), axis=1)
        
        # Escribir el frame combinado en el nuevo GIF
        writer.append_data(combined_frame)
    
    # Cerrar el escritor
    writer.close()
