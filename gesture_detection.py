import time

import cv2
import mediapipe as mp
import joblib
import numpy as np
import logging
from collections import deque, Counter

SECUENCIA_LONGITUD = 20  # Incrementado para mejor estabilidad


class GestureDetector:
    def __init__(self, gesture_model_path='modelo/modelo_gestos.joblib'):
        logging.basicConfig(level=logging.INFO)

        self.current_hand_landmarks = {'left': None, 'right': None}

        # Configuración de MediaPipe para detectar ambas manos
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Buffers separados para cada mano
        self.frame_buffer_left = []
        self.frame_buffer_right = []
        self.max_buffer_size = SECUENCIA_LONGITUD
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 0.5

        # Dimensiones de la imagen (ajustar según tu necesidad)
        self.image_width = 640
        self.image_height = 480

        try:
            self.pipeline = joblib.load(gesture_model_path)
            logging.info("Pipeline cargado exitosamente")
        except Exception as e:
            logging.error(f"Error cargando el pipeline: {e}")
            raise

    def _update_buffer(self, buffer, prediction):
        """Actualiza el buffer de predicciones manteniendo el tamaño máximo."""
        buffer.append(prediction)
        if len(buffer) > self.max_buffer_size:
            buffer.pop(0)
        return buffer

    def detect_gesture(self, frame):
        current_time = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        detected_gestures = {'left': None, 'right': None}
        self.current_hand_landmarks = {'left': None, 'right': None}  # Resetear landmarks

        try:
            if results.multi_hand_landmarks:
                for idx, (hand_landmarks, handedness) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)):

                    # Determinar si es mano izquierda o derecha
                    hand_type = 'left' if handedness.classification[0].label == 'Left' else 'right'

                    # Guardar los landmarks actuales
                    self.current_hand_landmarks[hand_type] = hand_landmarks

                    # Dibujar landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                    if current_time - self.last_prediction_time >= self.prediction_cooldown:
                        features = self._extract_hand_features(hand_landmarks)

                        if features is not None:
                            prediction = self.predict_gesture(features)

                            if prediction:
                                if hand_type == 'left':
                                    self.frame_buffer_left = self._update_buffer(
                                        self.frame_buffer_left, prediction)
                                    detected_gestures['left'] = self._get_stable_gesture(
                                        self.frame_buffer_left)
                                else:
                                    self.frame_buffer_right = self._update_buffer(
                                        self.frame_buffer_right, prediction)
                                    detected_gestures['right'] = self._get_stable_gesture(
                                        self.frame_buffer_right)

                                self.last_prediction_time = current_time

        except Exception as e:
            logging.error(f"Error en detección de gestos: {e}")

        return frame, detected_gestures

    def get_hand_landmarks(self, hand_type):
        """Obtiene los landmarks actuales para una mano específica."""
        return self.current_hand_landmarks.get(hand_type)

    def _get_stable_gesture(self, buffer):
        """Determina el gesto estable basado en las últimas predicciones."""
        if not buffer or len(buffer) < self.max_buffer_size // 2:
            return None

        # Contar ocurrencias de cada gesto
        gesture_counts = {}
        for gesture in buffer:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        # Encontrar el gesto más común
        max_count = 0
        stable_gesture = None
        for gesture, count in gesture_counts.items():
            if count > max_count:
                max_count = count
                stable_gesture = gesture

        # Verificar si el gesto es suficientemente estable (60% de concordancia)
        if max_count / len(buffer) >= 0.6:
            return stable_gesture
        return None

    def _normalize_coordinates(self, coords):
        """Normaliza las coordenadas usando el mismo método que en el dataset."""
        range_ = np.ptp(coords)
        if range_ == 0:
            return np.zeros_like(coords)
        return (coords - np.min(coords)) / (range_ + 1e-6)

    def _extract_hand_features(self, hand_landmarks):
        """
        Extrae características de los landmarks de la mano.
        """
        try:
            # Extraer coordenadas x, y, z de los landmarks
            x_ = []
            y_ = []
            z_ = []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                z_.append(landmark.z)

            # Convertir a numpy arrays
            x_ = np.array(x_)
            y_ = np.array(y_)
            z_ = np.array(z_)

            # Normalizar coordenadas
            x_norm = self._normalize_coordinates(x_)
            y_norm = self._normalize_coordinates(y_)
            z_norm = self._normalize_coordinates(z_)

            # Concatenar características
            features = np.concatenate([x_norm, y_norm, z_norm])

            # Validar características
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logging.warning("Características inválidas detectadas")
                return None

            if len(features) != 63:  # 21 landmarks * 3 coordenadas
                logging.warning(f"Número incorrecto de características: {len(features)}")
                return None

            return features

        except Exception as e:
            logging.error(f"Error en extracción de características: {e}")
            return None

    def predict_gesture(self, features):
        """
        Realiza la predicción usando el pipeline completo.
        """
        try:
            if features is None:
                return None

            # Reshape para la predicción
            features = features.reshape(1, -1)

            # Realizar predicción usando el pipeline completo
            prediction = self.pipeline.predict(features)[0]
            logging.debug(f"Predicción realizada: {prediction}")

            return prediction

        except Exception as e:
            logging.error(f"Error en predicción: {e}")
            return None