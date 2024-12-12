import pickle
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import warnings
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    class_weight='balanced',
                    random_state=42
                ),
                'needs_scaling': True
            },
            'SVM': {
                'model': SVC(
                    kernel='rbf',
                    probability=True,
                    class_weight='balanced',
                    random_state=42
                ),
                'needs_scaling': True
            }
        }
        self.scaler = StandardScaler()

    def debug_data(self, X, name=""):
        """Función de debug para inspeccionar los datos."""
        self.logger.info(f"Debug de datos {name}:")
        self.logger.info(f"Tipo de datos: {X.dtype}")
        self.logger.info(f"Forma: {X.shape}")
        self.logger.info(f"Rango de valores: [{np.min(X)}, {np.max(X)}]")
        self.logger.info(f"Media: {np.mean(X)}")
        self.logger.info(f"Desviación estándar: {np.std(X)}")

    def load_data(self, filename='modelo/data_gestos.pickle'):
        """
        Carga los datos del archivo pickle.

        Args:
            filename (str): Ruta al archivo de datos

        Returns:
            tuple: (X, y) datos y etiquetas
        """
        try:
            if not Path(filename).exists():
                self.logger.error(f"El archivo {filename} no existe")
                return None, None

            with open(filename, 'rb') as f:
                data_dict = pickle.load(f)

            if not isinstance(data_dict, dict) or 'data' not in data_dict or 'labels' not in data_dict:
                self.logger.error("Formato de datos incorrecto")
                return None, None

            X = np.array(data_dict['data'])
            y = np.array(data_dict['labels'])

            # Debug de datos cargados
            self.debug_data(X, "datos cargados")

            self.logger.info(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
            self.logger.info(f"Clases únicas: {np.unique(y)}")

            return X, y

        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            return None, None

    def verify_data(self, X, y):
        """Verifica la integridad de los datos antes del entrenamiento."""
        try:
            # Convertir a numpy array si no lo es
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            # Forzar la conversión a float64
            X = X.astype(np.float64, copy=True)

            # Verificar y limpiar datos
            mask_nan = np.isnan(X)
            mask_inf = np.isinf(X)
            if np.any(mask_nan) or np.any(mask_inf):
                self.logger.warning("Se encontraron valores NaN o Inf. Limpiando datos...")
                X[mask_nan | mask_inf] = 0.0

            # Verificar dimensiones
            if X.shape[0] != y.shape[0]:
                raise ValueError("Número de muestras no coincide entre X e y")

            # Verificar varianza
            variances = np.var(X, axis=0)
            if np.any(variances == 0):
                self.logger.warning("Algunas características tienen varianza cero")
                # Agregar pequeño ruido a características con varianza cero
                zero_var_cols = np.where(variances == 0)[0]
                X[:, zero_var_cols] += np.random.normal(0, 1e-5, size=(X.shape[0], len(zero_var_cols)))

            # Normalizar datos
            X = np.clip(X, -1e10, 1e10)
            max_abs_scaler = np.max(np.abs(X))
            if max_abs_scaler > 0:
                X = X / max_abs_scaler

            return X, y

        except Exception as e:
            self.logger.error(f"Error en verificación de datos: {e}")
            return None, None

    def evaluate_model(self, model, X_test, y_test):
        """Evalúa el rendimiento de un modelo."""
        try:
            y_pred = model.predict(X_test)
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        except Exception as e:
            self.logger.error(f"Error en evaluación: {e}")
            return None

    def train_models(self, X, y):
        """Entrena múltiples modelos y evalúa su rendimiento."""
        results = {}

        try:
            # Verificar datos
            X, y = self.verify_data(X, y)
            if X is None or y is None:
                return None

            # División de datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                stratify=y,
                random_state=42
            )

            # Escalar datos
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Debug de datos escalados
            self.debug_data(X_train_scaled, "datos escalados")

            # Validación cruzada estratificada
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for name, model_info in self.models.items():
                self.logger.info(f"\nEntrenando modelo: {name}")

                try:
                    model = model_info['model']

                    # Usar datos escalados si el modelo lo requiere
                    X_train_model = X_train_scaled if model_info['needs_scaling'] else X_train
                    X_test_model = X_test_scaled if model_info['needs_scaling'] else X_test

                    # Entrenamiento
                    model.fit(X_train_model, y_train)

                    # Evaluación
                    metrics = self.evaluate_model(model, X_test_model, y_test)

                    if metrics is None:
                        continue

                    # Validación cruzada
                    cv_scores = []
                    for train_idx, val_idx in skf.split(X, y):
                        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                        if model_info['needs_scaling']:
                            X_fold_train = self.scaler.fit_transform(X_fold_train)
                            X_fold_val = self.scaler.transform(X_fold_val)

                        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                        model.fit(X_fold_train, y_fold_train)
                        score = model.score(X_fold_val, y_fold_val)
                        cv_scores.append(score)

                    results[name] = {
                        'model': Pipeline([
                            ('scaler', self.scaler) if model_info['needs_scaling'] else ('passthrough', None),
                            ('classifier', model)
                        ]),
                        'metrics': metrics,
                        'cv_mean': np.mean(cv_scores),
                        'cv_std': np.std(cv_scores)
                    }

                    self.logger.info(f"Resultados para {name}:")
                    self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
                    self.logger.info(f"CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
                    self.logger.info("\nClassification Report:")
                    self.logger.info(metrics['classification_report'])

                except Exception as e:
                    self.logger.error(f"Error entrenando {name}: {e}")
                    continue

            return results

        except Exception as e:
            self.logger.error(f"Error en entrenamiento: {e}")
            return None

    def save_best_model(self, results, filename='modelo_gestos.joblib'):
        """Guarda el mejor modelo."""
        try:
            if not results:
                self.logger.error("No hay resultados para guardar")
                return

            # Seleccionar mejor modelo basado en accuracy
            best_model_name = max(results.keys(),
                                  key=lambda k: results[k]['metrics']['accuracy'])
            best_model = results[best_model_name]['model']

            # Guardar modelo
            joblib.dump(best_model, filename)
            self.logger.info(f"Mejor modelo ({best_model_name}) guardado en {filename}")

        except Exception as e:
            self.logger.error(f"Error guardando modelo: {e}")


def main():
    try:
        # Suprimir advertencias específicas
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        trainer = ModelTrainer()

        # Cargar datos
        X, y = trainer.load_data()

        if X is not None and y is not None:
            trainer.logger.info(f"Datos cargados: {X.shape[0]} muestras")

            # Entrenar modelos
            results = trainer.train_models(X, y)

            # Guardar mejor modelo
            if results:
                trainer.save_best_model(results)

    except Exception as e:
        logging.error(f"Error en el entrenamiento: {e}")


if __name__ == "__main__":
    main()
