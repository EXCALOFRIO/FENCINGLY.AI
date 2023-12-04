import os
import json
import numpy as np

def filter_poses(frame_data, min_points=15, dispersion_threshold=45):
    filtered_poses = []

    for persona in frame_data["people"]:
        keypoints = persona["pose_keypoints_2d"]
        x_coords = keypoints[0::3]
        y_coords = keypoints[1::3]
        confianza_coords = keypoints[2::3]

        # Filter points with confidence greater than 0.2
        puntos_filtrados = [(x, y) for x, y, confianza in zip(x_coords, y_coords, confianza_coords) if confianza > 0.2]

        # Check if there are enough points and average dispersion is greater than the threshold
        if len(puntos_filtrados) >= min_points:
            dispersión_media_persona = np.std(puntos_filtrados, axis=0).mean()
            if dispersión_media_persona > dispersion_threshold:
                filtered_poses.append(persona)

    frame_data["people"] = filtered_poses
    return frame_data

def process_json_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for archivo_json in sorted(os.listdir(input_folder)):
        if archivo_json.endswith(".json"):
            input_path = os.path.join(input_folder, archivo_json)
            output_path = os.path.join(output_folder, archivo_json)

            with open(input_path, 'r') as f:
                datos = json.load(f)

            filtered_data = filter_poses(datos, min_points=15, dispersion_threshold=45)

            with open(output_path, 'w') as f:
                json.dump(filtered_data, f, indent=2)

if __name__ == "__main__":
    input_folder_path = r"D:\Users\Alejandro\Downloads\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended (2)\openpose\output_json_folder\AbsoluteConventionalArrowcrab"  # Replace with the actual input folder path
    output_folder_path = r"D:\Users\Alejandro\Downloads\outp1"  # Replace with the desired output folder path

    process_json_files(input_folder_path, output_folder_path)
