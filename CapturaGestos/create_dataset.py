import cv2
import os
import mediapipe as mp
import pickle
import json
import numpy as np
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


class DatasetCreator:
    def __init__(self):
        """Inicializa la clase DatasetCreator."""
        self.logger = logging.getLogger(__name__)
        self.data = []
        self.labels = []
        self.gestos_count = {}
        self.MIN_SAMPLES_PER_CLASS = 50  # Número mínimo de muestras por clase

    def validate_features(self, features):
        """Valida la calidad de las características extraídas."""
        if features is None:
            return False
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return False
        if len(features) != 63:  # 21 landmarks x 3 coordenadas
            return False
        return True

    def extract_hand_landmarks(self, landmarks):
        """Extrae y normaliza características de landmarks."""
        try:
            # Extraer coordenadas
            x_ = np.array([landmark['x'] for landmark in landmarks])
            y_ = np.array([landmark['y'] for landmark in landmarks])
            z_ = np.array([landmark['z'] for landmark in landmarks])

            # Normalización robusta
            def normalize_coordinates(coords):
                range_ = np.ptp(coords)
                if range_ == 0:
                    return np.zeros_like(coords)
                return (coords - np.min(coords)) / (range_ + 1e-6)

            # Normalizamos las coordenadas
            x_norm = normalize_coordinates(x_)
            y_norm = normalize_coordinates(y_)
            z_norm = normalize_coordinates(z_)

            # Concatenar las coordenadas normalizadas
            features = np.concatenate([x_norm, y_norm, z_norm])

            # Validar las características extraídas
            if not self.validate_features(features):
                return None

            return features

        except Exception as e:
            self.logger.error(f"Error en extracción de características: {e}")
            return None

    def process_gesture_data(self, palabra, timestamp_path):
        """Procesa los datos de un gesto específico."""
        json_files = list(timestamp_path.glob('*landmarks.json'))

        for file in json_files:
            try:
                with open(file, 'r') as f:
                    frame_data = json.load(f)

                if frame_data and len(frame_data) > 0:
                    hand_data = frame_data[0]  # Primera mano detectada
                    features = self.extract_hand_landmarks(hand_data['landmarks'])

                    if features is not None:
                        self.data.append(features)
                        self.labels.append(palabra)
                        self.gestos_count[palabra] = self.gestos_count.get(palabra, 0) + 1

            except Exception as e:
                self.logger.error(f"Error procesando {file}: {e}")

    def create_dataset(self, data_dir='gestos'):
        """Crea el dataset a partir de los archivos de gestos."""
        data_path = Path(data_dir)

        if not data_path.exists():
            self.logger.error(f"El directorio {data_dir} no existe")
            return None

        # Procesar cada palabra (gesto)
        for palabra in data_path.iterdir():
            if not palabra.is_dir():
                continue

            self.logger.info(f"Procesando gesto: {palabra.name}")

            for timestamp_folder in palabra.iterdir():
                if timestamp_folder.is_dir():
                    self.process_gesture_data(palabra.name, timestamp_folder)

        # Validar cantidad mínima de muestras por clase
        for gesto, count in self.gestos_count.items():
            if count < self.MIN_SAMPLES_PER_CLASS:
                self.logger.warning(
                    f"Advertencia: {gesto} tiene {count} muestras "
                    f"(mínimo recomendado: {self.MIN_SAMPLES_PER_CLASS})"
                )

        return {
            'data': np.array(self.data),
            'labels': np.array(self.labels),
            'gestos_count': self.gestos_count
        }

    def save_dataset(self, dataset, filename='data_gestos.pickle'):
        """Guarda el dataset en un archivo."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(dataset, f)
            self.logger.info(f"Dataset guardado en {filename}")
        except Exception as e:
            self.logger.error(f"Error guardando dataset: {e}")


def main():
    try:
        creator = DatasetCreator()

        # Crear dataset
        dataset = creator.create_dataset()

        if dataset is not None:
            # Información del dataset
            n_samples = len(dataset['data'])
            n_classes = len(set(dataset['labels']))

            creator.logger.info(f"Dataset creado exitosamente:")
            creator.logger.info(f"Total de muestras: {n_samples}")
            creator.logger.info(f"Número de clases: {n_classes}")
            creator.logger.info("Distribución de clases:")

            for gesto, count in dataset['gestos_count'].items():
                creator.logger.info(f"  - {gesto}: {count} muestras")

            # Guardar dataset
            creator.save_dataset(dataset)

    except Exception as e:
        logging.error(f"Error en la creación del dataset: {e}")


if __name__ == "__main__":
    main()
