import cv2
import os
import mediapipe as mp
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


class CapturaLenguaSenas:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.logger = logging.getLogger(__name__)

    def crear_directorio(self, base_path, palabra):
        try:
            path = os.path.join(base_path, palabra)
            os.makedirs(path, exist_ok=True)
            return path
        except Exception as e:
            self.logger.error(f"Error creando directorio: {e}")
            return None

    def capturar_gesto(self, palabra, duracion=5):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.logger.error("No se pudo abrir la cámara")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        carpeta = self.crear_directorio('../gestos', palabra)

        if not carpeta:
            return

        carpeta = os.path.join(carpeta, timestamp)
        os.makedirs(carpeta, exist_ok=True)

        frames_grabados = 0
        inicio = cv2.getTickCount()

        while True:
            ret, frame = cap.read()
            if not ret:
                self.logger.warning("No se pudo capturar frame")
                break

            height, width, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                frame_data = []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[i].classification[0].label
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    hand_data = {
                        'label': hand_label,
                        'landmarks': []
                    }
                    for landmark in hand_landmarks.landmark:
                        hand_data['landmarks'].append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    frame_data.append(hand_data)

                # Guardar datos de landmarks en JSON
                with open(os.path.join(carpeta, f'frame_{frames_grabados}_landmarks.json'), 'w') as f:
                    json.dump(frame_data, f, indent=4)

            # Texto en pantalla
            cv2.putText(frame,
                        f"Grabando: {palabra} ({duracion - int(frames_grabados / 30)}s)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.imshow('Captura de Gesto', frame)
            cv2.imwrite(os.path.join(carpeta, f'frame_{frames_grabados}.jpg'), frame)
            frames_grabados += 1

            tiempo_actual = (cv2.getTickCount() - inicio) / cv2.getTickFrequency()
            if tiempo_actual >= duracion:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Grabación completada para: {palabra}")


def main():
    capturador = CapturaLenguaSenas()

    while True:
        print("\nCaptura de gestos para palabras/frases")
        print("1. Capturar un gesto")
        print("0. Salir")

        opcion = input("Ingrese opción: ")

        if opcion == '0':
            print("Saliendo...")
            break

        if opcion == '1':
            palabra = input("Ingrese la palabra o frase para capturar: ").strip()
            if palabra:
                try:
                    duracion = int(input("Duración de la captura (en segundos): "))
                    capturador.capturar_gesto(palabra, duracion=duracion)
                except ValueError:
                    print("Ingrese una duración válida en segundos")
                except Exception as e:
                    print(f"Error durante la captura: {e}")
            else:
                print("Debe ingresar una palabra o frase válida.")
        else:
            print("Opción no válida. Intente de nuevo.")


if __name__ == "__main__":
    main()
