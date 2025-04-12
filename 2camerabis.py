from ultralytics import YOLO  # Importer YOLO
import cv2
import numpy as np
import time
# Déclaration du dictionnaire ArUco
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50
}
# Charger le modèle YOLO (remplacez par votre propre modèle)
model = YOLO("seg_planche(small).pt")  # Chemin vers vos poids YOLO
def calculate_custom_axes_from_tag(rvec, tvec):
    """
    Utilise les axes d'un marqueur ArUco (rvec, tvec) pour définir la matrice de rotation et la position de l'origine.
    :param rvec: Vecteur de rotation du marqueur (3x1).
    :param tvec: Vecteur de translation du marqueur (3x1).
    :return: Matrice de rotation (3x3) et position de l'origine (3x1).
    """
    # Convertir le vecteur de rotation en une matrice de rotation
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # La position de l'origine est directement donnée par tvec
    origin_center = tvec.flatten()  # Assure que c'est un vecteur 1D

    print(f"Matrice de rotation obtenue directement :\n{rotation_matrix}")
    print(f"Centre de l'origine (tvec) : {origin_center}")
    return rotation_matrix, origin_center

def detect_and_draw_planches(frame, roi):
    """
    Applique YOLO à une région d'intérêt (ROI) pour détecter des planches et dessine leurs contours.
    """
    x1, y1, x2, y2 = roi

    # Vérifier les dimensions de la ROI
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        print(f"ROI invalide : {roi}")
        return False

    # Extraire la région d'intérêt
    cropped_frame = frame[y1:y2, x1:x2]

    # Vérifier que l'image extraite n'est pas vide
    if cropped_frame.size == 0:
        print("ROI vide après extraction.")
        return False

    # Appliquer YOLO pour détecter les objets dans la ROI
    results = model(cropped_frame)

    detected = False
    for result in results[0].boxes.data:
        class_id, confidence, x_min, y_min, x_max, y_max = int(result[5]), result[4], *result[:4]
        if class_id == 0 and confidence > 0.5:  # Classe 0 : planches (modifiez selon votre dataset)
            detected = True

            # Convertir les coordonnées locales (dans la ROI) en coordonnées globales
            x_min_global = int(x1 + x_min)
            y_min_global = int(y1 + y_min)
            x_max_global = int(x1 + x_max)
            y_max_global = int(y1 + y_max)

            # Dessiner un contour autour de la planche détectée
            cv2.rectangle(frame, (x_min_global, y_min_global), (x_max_global, y_max_global), (0, 255, 0), 2)  # Vert pour la planche

    return detected


def draw_red_square_custom_axes(frame, rotation_matrix, tvec, matrix_coefficients, distortion_coefficients, square_position, square_side):
    """
    Dessine un carré rouge dans un plan défini par des axes personnalisés, avec vérification des planches.
    """
    if rotation_matrix is None or tvec is None:
        print("Erreur : Matrice de rotation ou vecteur de translation invalide.")
        return

    x, y = square_position
    side = square_side
    square_points_3D = np.array([
        [x, y, 0.11],
        [x + 3*side, y, 0.11],
        [x + 3*side, y + side, 0.11],
        [x, y + side, 0.11]
    ], dtype=np.float32)

    square_points_2D, _ = cv2.projectPoints(
        square_points_3D, rotation_matrix, tvec, matrix_coefficients, distortion_coefficients
    )
    square_points_2D = np.int32(square_points_2D).reshape(-1, 2)

    # Calcul des coordonnées de la ROI dans l'image
    x1, y1 = np.min(square_points_2D, axis=0)
    x2, y2 = np.max(square_points_2D, axis=0)

    # Appliquer la détection YOLO sur la ROI et dessiner les contours
    if detect_and_draw_planches(frame, (x1, y1, x2, y2)):
        cv2.polylines(frame, [square_points_2D], isClosed=True, color=(0, 255, 0), thickness=2)  # Vert si planche détectée
    else:
        cv2.polylines(frame, [square_points_2D], isClosed=True, color=(0, 0, 255), thickness=2)  # Rouge sinon
        
def draw_blue_square_custom_axes(frame, rotation_matrix, tvec, matrix_coefficients, distortion_coefficients, square_position, square_side):
    if rotation_matrix is None or tvec is None:
        print("Erreur : Matrice de rotation ou vecteur de translation invalide.")
        return

    x, y = square_position
    side = square_side
    square_points_3D = np.array([
        [x, y, 0.11],
        [x , y+ 3*side, 0.11],
        [x + side, y + 3*side, 0.11],
        [x+ side, y , 0.11]
    ], dtype=np.float32)

    square_points_2D, _ = cv2.projectPoints(
        square_points_3D, rotation_matrix, tvec, matrix_coefficients, distortion_coefficients
    )
    square_points_2D = np.int32(square_points_2D).reshape(-1, 2)

    # Calcul des coordonnées de la ROI dans l'image
    x1, y1 = np.min(square_points_2D, axis=0)
    x2, y2 = np.max(square_points_2D, axis=0)

    # Appliquer la détection YOLO sur la ROI et dessiner les contours
    if detect_and_draw_planches(frame, (x1, y1, x2, y2)):
        cv2.polylines(frame, [square_points_2D], isClosed=True, color=(0, 255, 0), thickness=2)  # Vert si planche détectée
    else:
        cv2.polylines(frame, [square_points_2D], isClosed=True, color=(0, 0, 255), thickness=2)  # Rouge sinon
        

def draw_origin_on_marker(frame, center):
    """
    Dessine un cercle rouge pour représenter l'origine sur le marqueur.
    """
    if center is None:
        print("Erreur : Centre du marqueur invalide.")
        return

    # Dessiner un cercle rouge au centre du marqueur
    center_coords = (int(center[0]), int(center[1]))
    cv2.circle(frame, center_coords, radius=5, color=(0, 0, 255), thickness=-1)  # Cercle plein
    print(f"Origine dessinée au centre du marqueur : {center_coords}")

def pose_estimation_with_custom_axes_and_origin(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    """
    Détecte deux marqueurs ArUco, dessine un cercle rouge sur le marqueur en haut
    pour représenter l'origine, et dessine un carré rouge dans le plan 3D défini.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    # Détecter les marqueurs
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and len(corners) >= 2:
        # Identifier le marqueur supérieur droit comme origine
        origin_index = np.argmin([np.sum(np.mean(corner[0], axis=0)) for corner in corners])

        # Obtenir rvec et tvec pour tous les marqueurs
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.1, matrix_coefficients, distortion_coefficients)

        # Obtenir rvec et tvec pour le marqueur supérieur droit (origine)
        rvec_origin = rvecs[origin_index]
        tvec_origin = tvecs[origin_index]

        # Dessiner l'origine
        origin_center = np.mean(corners[origin_index][0], axis=0)
        draw_origin_on_marker(frame, origin_center)

        # Dessiner les axes 3D des marqueurs
        for i in range(len(corners)):
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvecs[i], tvecs[i], 0.1)

        # Dessiner un carré rouge dans le plan défini par l'origine
        square_position = (-0.10, -0.50)   # Position du carré en mètres
        square_side = 0.17  # Taille du carré en mètres
        draw_red_square_custom_axes(frame, rvec_origin, tvec_origin, matrix_coefficients, distortion_coefficients, square_position, square_side)
# Dessiner un rectangle dans le plan défini par l'origine
        square_position = (0.40, -0.35)  # Position du carré en mètres
        square_side = 0.15  # Taille du carré en mètres
        draw_blue_square_custom_axes(frame, rvec_origin, tvec_origin, matrix_coefficients, distortion_coefficients, square_position, square_side)

        # Dessiner les contours des marqueurs détectés
        cv2.aruco.drawDetectedMarkers(frame, corners)
    else:
        print("Erreur : Moins de deux marqueurs détectés.")

    return frame

if __name__ == '__main__':
    k = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)  # Matrice intrinsèque
    d = np.zeros((5, 1))  # Coefficients de distorsion

    aruco_dict_type = ARUCO_DICT["DICT_4X4_50"]

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Estimation de pose avec origine et carré rouge
        output = pose_estimation_with_custom_axes_and_origin(frame, aruco_dict_type, k, d)
        cv2.imshow('Pose Estimation - Origin and Custom Axes', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

