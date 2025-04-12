import cv2
import numpy as np
import sys
import argparse
import time
import OUI

# Déclaration de ARUCO_DICT en utilisant cv2.aruco
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50
}
def draw_square_near_tag(frame, rvec, tvec, matrix_coefficients, distortion_coefficients):
    # Taille du tag et du carré
    tag_size = 0.05  # 5 cm en mètres
    square_size = 0.10  # 10 cm en mètres
    offset = 0.05  # 5 cm en mètres

    # Points 3D du carré dans le repère du marqueur
    square_points_3D = np.array([
        [tag_size + offset, -square_size / 2, 0],
        [tag_size + offset, square_size / 2, 0],
        [tag_size + offset + square_size, square_size / 2, 0],
        [tag_size + offset + square_size, -square_size / 2, 0]
    ], dtype=np.float32)

    # Projeter les points 3D sur l'image 2D
    square_points_2D, _ = cv2.projectPoints(
        square_points_3D, rvec, tvec, matrix_coefficients, distortion_coefficients)

    # Convertir les points en entiers et dessiner le carré
    square_points_2D = np.int32(square_points_2D).reshape(-1, 2)
    cv2.polylines(frame, [square_points_2D], isClosed=True, color=(255, 0, 0), thickness=2)
def get_marker_center(corner):
    # Calcule le centre d'un marqueur ArUco à partir de ses coins
    x = int(np.mean(corner[:, 0]))
    y = int(np.mean(corner[:, 1]))
    return (x, y)

def sort_markers_clockwise(corners):
    # Calcule le centre de chaque marqueur
    centers = [get_marker_center(corner[0]) for corner in corners]

    if len(centers) != 4:
        return None  # Retourner None si le nombre de marqueurs n'est pas 4

    # Calculer le centre de masse de tous les marqueurs
    center_of_mass = np.mean(centers, axis=0)

    # Calculer l'angle de chaque marqueur par rapport au centre de masse
    angles = [np.arctan2(center[1] - center_of_mass[1], center[0] - center_of_mass[0]) for center in centers]

    # Trier les marqueurs par angle dans le sens horaire
    sorted_indices = np.argsort(angles)
    ordered_markers = [centers[i] for i in sorted_indices]

    # Identifier le marqueur en bas à gauche (celui avec les plus petites coordonnées y, puis x)
    bottom_left_index = np.argmin([center[1] + center[0] for center in ordered_markers])

    # Réorganiser les marqueurs pour que le premier soit celui en bas à gauche
    ordered_markers = ordered_markers[bottom_left_index:] + ordered_markers[:bottom_left_index]

    return ordered_markers

def calculate_angle_normal_to_camera(rvec):
    # Convertir le vecteur de rotation en une matrice de rotation
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    normal_vector = np.dot(rotation_matrix, np.array([0, 0, 1]))
    angle = np.arccos(np.clip(normal_vector[2], -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    return angle_degrees

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    angles = []  # Liste pour stocker les angles de chaque marqueur
    # Si des marqueurs sont détectés
    if ids is not None and len(corners) > 0:
        ordered_markers = sort_markers_clockwise(corners)
        if ordered_markers is not None:  # Vérifier que le tri a réussi
            # Numéroter les marqueurs de 1 à 4
            for i, center in enumerate(ordered_markers):
                cv2.putText(frame, f"{i+1}", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
             # Estimer la pose de chaque marqueur et retourner les valeurs rvec et tvec
                rvec, tvec, markerpoints = cv2.aruco.estimatePoseSingleMarkers(
                 corners[i], 0.02, matrix_coefficients, distortion_coefficients)
                print(tvec)
                 # Dessiner un carré autour des marqueurs
                cv2.aruco.drawDetectedMarkers(frame, corners)
        
                # Dessiner l'axe (assurez-vous que rvec et tvec sont bien indexés)
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
                  # Calculer l'angle
                angle = calculate_angle(rvec[0])
                angles.append(angle)
                # Afficher l'angle sur l'image
                text = f"Angle: {angle:.2f} deg"
                position = (int(corners[i][0][0][0]), int(corners[i][0][0][1] - 10))
                cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
           # Dessiner le carré de 10 cm de côté à 5 cm du tag
           # draw_square_near_tag(frame, rvec[0], tvec[0], matrix_coefficients, distortion_coefficients)
        
            
      

        # Vérifier si les angles sont proches les uns des autres
        if len(angles) == 4:
            max_angle = max(angles)
            min_angle = min(angles)
            difference = max_angle - min_angle
            print(f"Différence maximale entre les angles : {difference:.2f} degrés")

            
    return frame
  
def calculate_angle(rvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    normal_vector = np.dot(rotation_matrix, np.array([0, 0, 1]))
    camera_axis = np.array([0, 0, 1])
    dot_product = np.dot(normal_vector, camera_axis)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Limiter la plage pour éviter les erreurs numériques
    angle_degrees = np.degrees(angle)
    return angle_degrees


if __name__ == '__main__':
    # Pour les tests rapides, utilisez des matrices de caméra par défaut
    k = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    d = np.zeros((5, 1))  # Valeurs par défaut pour les coefficients de distorsion

    aruco_dict_type = ARUCO_DICT["DICT_4X4_50"]

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        output = pose_estimation(frame, aruco_dict_type, k, d)
        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


    """Ici j'étudie les différences d'angles entre chaque marqueurs que je peux observer avec pour savoir si c'est fiable ou pas"""