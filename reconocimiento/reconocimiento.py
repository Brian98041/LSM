import cv2
import mediapipe as mp
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sklearn.svm import SVC
from utilidades import normalizar_keypoints, calcular_angulos, calcular_distancias

# Inicializar MediaPipe para Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def cargar_gestures():
    """
    Recorre la carpeta 'datasets' y carga los datos de cada seña (archivo JSON)
    para formar el dataset de entrenamiento.
    """
    gestures_data = []
    base_folder = "./datasets"
    
    if not os.path.exists(base_folder):
        print("La carpeta 'datasets' no existe.")
        return gestures_data

    # Cada subcarpeta corresponde a una seña (el nombre de la carpeta es el nombre de la seña)
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            sign_name = folder
            # Iterar sobre cada archivo JSON en la carpeta de la seña
            for file in os.listdir(folder_path):
                if file.endswith(".json"):
                    json_path = os.path.join(folder_path, file)
                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)
                            # Se espera que el JSON tenga las claves "keypoints", "angles" y "distances"
                            if "keypoints" in data and "angles" in data and "distances" in data:
                                gestures_data.append({
                                    "sign_name": sign_name,
                                    "keypoints": data["keypoints"],
                                    "angles": data["angles"],
                                    "distances": data["distances"]
                                })
                            else:
                                print(f"Advertencia: Faltan datos en {json_path}")
                    except Exception as e:
                        print(f"Error al leer {json_path}: {e}")
    return gestures_data

def entrenar_clasificador(gestures_data):
    """
    Prepara las entradas concatenando los keypoints, ángulos y distancias y entrena un clasificador SVM.
    """
    X = []
    y = []

    for gesture in gestures_data:
        sign_name = gesture["sign_name"]
        keypoints = gesture["keypoints"]
        angles = gesture["angles"]

        # Aunque los keypoints ya fueron normalizados al capturar, se aplica la normalización
        # por consistencia. (Si ya están normalizados, la transformación es neutra).
        keypoints_normalizados = normalizar_keypoints(keypoints)
        # Aplanar la lista de keypoints para obtener un vector unidimensional
        keypoints_flat = [value for kp in keypoints_normalizados for value in kp.values()]

        # Se recalculan las distancias a partir de los keypoints normalizados
        distancias = calcular_distancias(keypoints_normalizados)

        # Concatenar la información en una sola entrada
        entrada = keypoints_flat + angles + distancias

        X.append(entrada)
        y.append(sign_name)

    if len(X) == 0:
        raise ValueError("No hay datos disponibles para entrenar el modelo SVM.")

    svm = SVC(kernel="linear")
    svm.fit(X, y)
    return svm

def reconocer_seña(svm, keypoints_actuales):
    """
    Dado un conjunto de keypoints actuales, procesa la información y utiliza el clasificador SVM para reconocer la seña.
    """
    keypoints_normalizados = normalizar_keypoints(keypoints_actuales)
    angles_actuales = calcular_angulos(keypoints_normalizados)
    distancias_actuales = calcular_distancias(keypoints_normalizados)

    keypoints_flat = [value for kp in keypoints_normalizados for value in kp.values()]
    entrada = keypoints_flat + angles_actuales + distancias_actuales

    try:
        senia_reconocida = svm.predict([entrada])
        return senia_reconocida[0]
    except Exception as e:
        print(f"Error en reconocimiento: {e}")
        return "Seña no reconocida"

def reconocer_señas_en_tiempo_real():
    """
    Carga los datos de las señas desde la carpeta 'datasets', entrena el modelo SVM y realiza
    el reconocimiento en tiempo real utilizando la cámara.
    """
    gestures_data = cargar_gestures()
    
    if not gestures_data:
        print("No se encontraron señas en la carpeta 'datasets'.")
        return

    svm = entrenar_clasificador(gestures_data)
    #print("Modelo SVM entrenado exitosamente.")

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            results = hands.process(image_rgb)

            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    keypoints_actuales = []
                    for landmark in hand_landmarks.landmark:
                        keypoints_actuales.append({
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        })

                    if keypoints_actuales:
                        senia_reconocida = reconocer_seña(svm, keypoints_actuales)
                        if senia_reconocida:
                            cv2.putText(
                                image_bgr,
                                senia_reconocida,
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 0, 0),
                                2,
                                cv2.LINE_AA
                            )

            cv2.imshow("Reconocimiento de Señas", image_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reconocer_señas_en_tiempo_real()
