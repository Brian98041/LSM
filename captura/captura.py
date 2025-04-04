import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mediapipe as mp
import json
from utilidades import normalizar_keypoints, calcular_angulos, calcular_distancias

# Inicializar MediaPipe para Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def solicitar_datos_seña():
    etiqueta = input("Introduce el nombre de la seña (ej. 'Por favor'): ")
    categoria = input("Introduce la categoría de la seña (ej. 'Gestos comunes'): ")
    return etiqueta, categoria

def obtener_numero_imagen(carpeta):
    """
    Retorna el número de la siguiente imagen a guardar, basado en los archivos ya existentes en la carpeta.
    """
    archivos = [f for f in os.listdir(carpeta) if f.startswith("imagen_") and f.endswith(".jpg")]
    return len(archivos) + 1

def registrar_seña():
    etiqueta, categoria = solicitar_datos_seña()

    # Crear la carpeta del dataset para la seña, si no existe.
    carpeta = f"./datasets/{etiqueta}"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    cap = cv2.VideoCapture(0)
    capturando = False  # Bandera para iniciar/detener la captura

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
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })

                    if not keypoints_actuales:
                        print("No se detectaron keypoints.")
                        continue

                    keypoints_normalizados = normalizar_keypoints(keypoints_actuales)
                    angulos = calcular_angulos(keypoints_normalizados)
                    distancias = calcular_distancias(keypoints_normalizados)

                    h, w, _ = frame.shape
                    x_min = min([int(kp['x'] * w) for kp in keypoints_actuales])
                    x_max = max([int(kp['x'] * w) for kp in keypoints_actuales])
                    y_min = min([int(kp['y'] * h) for kp in keypoints_actuales])
                    y_max = max([int(kp['y'] * h) for kp in keypoints_actuales])

                    # Aumentar el margen alrededor de la mano
                    margen = 30  # Ajusta el valor para el margen deseado
                    x_min = max(x_min - margen, 0)
                    y_min = max(y_min - margen, 0)
                    x_max = min(x_max + margen, w)
                    y_max = min(y_max + margen, h)

                    recorte_manos = image_bgr[y_min:y_max, x_min:x_max]

                    if capturando:
                        # Generar el número de muestra en base a los archivos existentes.
                        sample_id = obtener_numero_imagen(carpeta)
                        ruta_imagen = f"{carpeta}/imagen_{sample_id}.jpg"
                        cv2.imwrite(ruta_imagen, recorte_manos)
                        print(f"Imagen recortada guardada en: {ruta_imagen}")

                        # Guardar los datos de keypoints, ángulos y distancias en un archivo JSON.
                        datos = {
                            "keypoints": keypoints_normalizados,
                            "angles": angulos,
                            "distances": distancias
                        }
                        ruta_json = f"{carpeta}/imagen_{sample_id}.json"
                        with open(ruta_json, "w") as f:
                            json.dump(datos, f)
                        print(f"Datos guardados en: {ruta_json}")

                        print("Captura completada.")
                        capturando = False

            cv2.imshow("Capturando seña fija - Presiona 's' para capturar, 'q' para salir", image_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                capturando = True
                print("Iniciando captura...")

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    registrar_seña()
