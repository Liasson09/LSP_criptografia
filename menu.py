import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

from cifrado_senas import CifradoSenas
from gesture_detection import GestureDetector
from traductor_senas import TraductorSenas
from cmu_decoder import CMUDecoder
import logging
import time

class SignLanguageApp:
    def __init__(self, root):
        # Configurar logger
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        self.cap = None  # Inicializar explícitamente
        self.root = root
        self.root.title("Sistema de Reconocimiento de Lengua de Señas Peruanas")
        self.root.configure(bg='#f0f0f0')

        # Configuración inicial de la ventana
        self.root.geometry("1000x700")
        self.root.resizable(False, False)

        # Estilo personalizado
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.setup_styles()

        try:
            # Inicializar componentes del sistema
            self.detector = GestureDetector()
            self.traductor = TraductorSenas()
            self.cmu_decoder = CMUDecoder()
            self.cifrador = CifradoSenas()
            self.logger.info("Componentes del sistema inicializados correctamente")
        except Exception as e:
            self.logger.error(f"Error inicializando componentes: {e}")
            messagebox.showerror("Error", f"Error inicializando el sistema: {str(e)}")

        # Variables de control
        self.recorded_gestures = []
        self.is_recording = False

        # Configuración de la interfaz gráfica
        self.setup_ui()

        # Configurar el protocolo de cierre de ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_styles(self):
        self.style.configure('TButton',
                             font=('Arial', 12, 'bold'),
                             padding=10)

        self.style.configure('TLabel',
                             font=('Arial', 12),
                             background='#f0f0f0')

        self.style.configure('TFrame',
                             background='#f0f0f0')

    def setup_ui(self):
        # Frame principal, centrado
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configuración de columnas y filas
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)

        # Frame izquierdo para botones
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=10)

        # Botones
        buttons = [
            ("Iniciar Grabación", self.start_recording),
            ("Detener Grabación", self.stop_recording),
            ("Cifrar Frase", self.show_cipher_menu),
            ("Mostrar Detalles", self.show_gesture_details)
        ]

        for text, command in buttons:
            btn = ttk.Button(left_frame, text=text, command=command)
            btn.pack(pady=10, fill='x')

        # Frame derecho
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=10)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=3)
        right_frame.rowconfigure(1, weight=1)

        # Área de video
        video_container = ttk.Frame(right_frame, borderwidth=2, relief='ridge')
        video_container.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
        video_container.grid_propagate(False)

        self.video_label = ttk.Label(video_container)
        self.video_label.pack(expand=True, fill='both', padx=5, pady=5)

        # Área de resultados
        result_frame = ttk.Frame(right_frame)
        result_frame.grid(row=1, column=0, sticky='nsew')

        scrollbar = ttk.Scrollbar(result_frame)
        scrollbar.pack(side='right', fill='y')

        self.result_text = tk.Text(
            result_frame,
            height=10,
            width=50,
            yscrollcommand=scrollbar.set,
            font=('Consolas', 10),
            bg='#f9f9f9',
            relief='flat',
            padx=10,
            pady=10
        )
        self.result_text.pack(side='left', expand=True, fill='both')
        scrollbar.config(command=self.result_text.yview)

    def start_recording(self):
        try:
            self.is_recording = True
            self.recorded_gestures = []
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Iniciando grabación...\n")

            if self.cap is None:
                self.cap = cv2.VideoCapture(0)

            self.capture_video()
        except Exception as e:
            self.logger.error(f"Error al iniciar grabación: {e}")
            messagebox.showerror("Error", "No se pudo iniciar la grabación")

    def capture_video(self):
        if not self.is_recording or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.logger.error("Error al capturar el frame del video.")
            self.stop_recording()
            return

        try:
            frame = cv2.flip(frame, 1)
            frame, gestures = self.detector.detect_gesture(frame)

            for hand_type, gesture in gestures.items():
                if gesture:
                    cmu_notation = self._obtener_notacion_cmu(gesture)
                    if cmu_notation:
                        hand_landmarks = self.detector.get_hand_landmarks(hand_type)
                        if hand_landmarks:
                            self._draw_hand_label(frame, hand_landmarks, hand_type, gesture, cmu_notation)

                        # Verificar si el gesto ya está registrado ANTES de agregarlo
                        if not any(g[0] == gesture for g in self.recorded_gestures):  # Evitar duplicados
                            self.recorded_gestures.append((gesture, cmu_notation))
                            self._mostrar_gesto_detectado(hand_type, gesture, cmu_notation)  # Mostrar SOLO si es nuevo

            # Mostrar información del gesto detectado (si hay alguno)
            current_gesture_info = None
            for hand_type, gesture in gestures.items():
                if gesture:
                    current_gesture_info = gesture
                    break  # Mostrar solo un gesto si se detectan varios

            self._add_info_overlay(frame, current_gesture_info)

            # Mostrar frame en la interfaz
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        except Exception as e:
            self.logger.error(f"Error en procesamiento de video: {e}")

        self.root.after(30, self.capture_video)

    def _mostrar_gesto_detectado(self, hand_type, gesture, cmu_notation):
        try:
            self.result_text.insert(tk.END, f"\n{'=' * 50}")
            self.result_text.insert(tk.END, f"\nGesto Detectado - Mano {hand_type.upper()}")
            self.result_text.insert(tk.END, f"\n{'=' * 50}\n")
            self.result_text.insert(tk.END, f"Gesto: {gesture}\n")

            # Obtener códigos CMU
            if cmu_notation:
                # Mostrar códigos CMU
                self.result_text.insert(tk.END, "\nCódigos CMU:")
                config_code = next(k for k, v in self.cmu_decoder.configuracion_labels.items()
                                   if v == cmu_notation['configuracion'])
                mov_code = next(k for k, v in self.cmu_decoder.movimiento_labels.items()
                                if v == cmu_notation['movimiento'])
                loc_code = next(k for k, v in self.cmu_decoder.ubicacion_labels.items()
                                if v == cmu_notation['ubicacion'])

                self.result_text.insert(tk.END, f"\n- Configuración: {config_code}")
                self.result_text.insert(tk.END, f"\n- Movimiento: {mov_code}")
                self.result_text.insert(tk.END, f"\n- Ubicación: {loc_code}")

                # Mostrar descripciones
                self.result_text.insert(tk.END, "\n\nDescripciones:")
                self.result_text.insert(tk.END, f"\n- Configuración: {cmu_notation['configuracion']}")
                self.result_text.insert(tk.END, f"\n- Movimiento: {cmu_notation['movimiento']}")
                self.result_text.insert(tk.END, f"\n- Ubicación: {cmu_notation['ubicacion']}")

            else:
                self.result_text.insert(tk.END, "\nNo se pudo obtener la notación CMU para este gesto.")

            self.result_text.insert(tk.END, f"\n{'=' * 50}\n")
            self.result_text.see(tk.END)

        except Exception as e:
            self.logger.error(f"Error mostrando gesto: {e}")
            self.result_text.insert(tk.END, "\nError al mostrar detalles del gesto.\n")


    def _add_info_overlay(self, frame, current_gesture):
        """Agrega overlay con información en el frame."""

        height, width = frame.shape[:2]

        # Agregar rectángulo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 70), (width - 10, height - 10),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Agregar texto
        status = "Grabando..." if self.is_recording else "Detenido"
        gesture_text = f"Gesto: {current_gesture}" if current_gesture else "No se detecta gesto"

        cv2.putText(frame, status, (20, height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, gesture_text, (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _obtener_notacion_cmu(self, gesture):
        """Obtiene la notación CMU para un gesto usando el CMUDecoder."""
        try:
            # Inicializar un diccionario de notación
            notacion = {
                'configuracion': None,
                'movimiento': None,
                'ubicacion': None
            }

            # Buscar coincidencias en cada categoría
            for categoria in ['configuracion', 'movimiento', 'ubicacion']:
                for codigo, descripcion in getattr(self.cmu_decoder, f"{categoria}_labels").items():
                    if gesture.lower() in descripcion.lower():
                        notacion[categoria] = descripcion
                        break

            # Verificar si se encontraron todas las categorías
            if all(notacion.values()):
                return notacion
            else:
                self.logger.warning(f"No se pudo encontrar notación completa para el gesto: {gesture}")
                return None

        except Exception as e:
            self.logger.error(f"Error obteniendo notación CMU: {e}")
            return None

    def show_cipher_menu(self):
        if not self.recorded_gestures:
            messagebox.showwarning("Advertencia", "No hay gestos grabados para cifrar.")
            return

        cipher_window = tk.Toplevel(self.root)
        cipher_window.title("Cifrado de Frase")
        cipher_window.geometry("300x200")

        ttk.Label(cipher_window, text="Seleccione método de cifrado:").pack(pady=10)

        method_var = tk.StringVar(value="simple")
        methods = [
            ("Cifrado Simple", "simple"),
            ("Cifrado Afín", "afin"),
            ("Cifrado Vigenère", "vigenere")
        ]

        for text, value in methods:
            ttk.Radiobutton(cipher_window, text=text, variable=method_var, value=value).pack()

        ttk.Button(cipher_window, text="Cifrar",
                   command=lambda: self.cipher_phrase(method_var.get())).pack(pady=20)

    def cipher_phrase(self, method):
        if not self.recorded_gestures:
            messagebox.showwarning("Advertencia", "No hay gestos grabados para cifrar.")
            return

        try:
            ciphered_gestures = []
            for gesture, notation in self.recorded_gestures:
                try:
                    notacion_numerica = self._notation_to_tuple(notation)
                    if notacion_numerica:  # Verificar si la conversión fue exitosa
                        notacion_cifrada_tupla = self.cifrador.cifrar(notacion_numerica, method)
                        notacion_cifrada = self._tuple_to_notation(
                            notacion_cifrada_tupla)  # Convertir de vuelta a texto

                        ciphered_gestures.append({
                            'gesto_original': gesture,
                            'notacion_original': notation,  # Usar la notación original directamente
                            'notacion_cifrada': notacion_cifrada  # Guardar la notación cifrada como texto
                        })
                    else:
                        self.logger.error(f"Error al convertir la notación a tupla para el gesto: {gesture}")

                except Exception as e:
                    self.logger.error(f"Error procesando gesto {gesture}: {e}")
                    continue

            # Mostrar resultados del cifrado
            if ciphered_gestures:
                self.result_text.insert(tk.END, f"\n=== Cifrado {method.capitalize()} ===\n")

                for cifrado in ciphered_gestures:
                    self.result_text.insert(tk.END,
                                            f"\nGesto: {cifrado['gesto_original']}\n"
                                            f"\nNotación Original:"
                                            f"\n- Configuración: {cifrado['notacion_original']['configuracion']}"
                                            f"\n- Movimiento: {cifrado['notacion_original']['movimiento']}"
                                            f"\n- Ubicación: {cifrado['notacion_original']['ubicacion']}"
                                            f"\n\nNotación Cifrada:"
                                            f"\n- Configuración: {cifrado['notacion_cifrada']['configuracion']}"
                                            f"\n- Movimiento: {cifrado['notacion_cifrada']['movimiento']}"
                                            f"\n- Ubicación: {cifrado['notacion_cifrada']['ubicacion']}\n"
                                            )

                self.result_text.see(tk.END)
            else:
                messagebox.showwarning("Advertencia", "No se pudo cifrar ningún gesto.")

        except Exception as e:
            error_msg = f"Error al cifrar: {str(e)}"
            self.logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def _notation_to_tuple(self, notation):
        """Convierte la notación de texto a una tupla numérica."""
        try:
            configuracion_num = next(
                (codigo for codigo, desc in self.cmu_decoder.configuracion_labels.items()
                 if desc == notation['configuracion']), None)
            movimiento_num = next(
                (codigo for codigo, desc in self.cmu_decoder.movimiento_labels.items()
                 if desc == notation['movimiento']), None)
            ubicacion_num = next(
                (codigo for codigo, desc in self.cmu_decoder.ubicacion_labels.items()
                 if desc == notation['ubicacion']), None)

            if all([configuracion_num, movimiento_num, ubicacion_num]):
                return (int(configuracion_num), int(movimiento_num), int(ubicacion_num))
            return None
        except Exception as e:
            self.logger.error(f"Error convirtiendo notación: {e}")
            return None

    def _tuple_to_notation(self, tupla):
        """Convierte una tupla numérica a notación de texto."""
        try:
            return {
                'configuracion': self.cmu_decoder.configuracion_labels.get(str(tupla[0]), 'Desconocido'),
                'movimiento': self.cmu_decoder.movimiento_labels.get(str(tupla[1]), 'Desconocido'),
                'ubicacion': self.cmu_decoder.ubicacion_labels.get(str(tupla[2]), 'Desconocido')
            }
        except Exception as e:
            self.logger.error(f"Error convirtiendo tupla a notación: {e}")
            return None

    def stop_recording(self):
        try:
            self.is_recording = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.result_text.insert(tk.END, "\nGrabación detenida.\n")
            self.video_label.configure(image='')
        except Exception as e:
            self.logger.error(f"Error al detener grabación: {e}")

    def show_gesture_details(self):
        if not self.recorded_gestures:
            messagebox.showwarning("Advertencia", "No hay gestos grabados para mostrar.")
            return

        self.result_text.insert(tk.END, "\n=== Detalles de Gestos ===\n")

        for gesture, notation in self.recorded_gestures:
            self.result_text.insert(tk.END, f"\nGesto: {gesture}\n")
            self.result_text.insert(tk.END, f"Configuración: {notation['configuracion']}\n")
            self.result_text.insert(tk.END, f"Movimiento: {notation['movimiento']}\n")
            self.result_text.insert(tk.END, f"Ubicación: {notation['ubicacion']}\n")

        self.result_text.see(tk.END)

    def on_closing(self):
        try:
            if self.cap is not None:
                self.cap.release()
            self.root.destroy()
        except Exception as e:
            self.logger.error(f"Error al cerrar la aplicación: {e}")

    def _draw_hand_label(self, frame, hand_landmarks, hand_type, gesture, cmu_notation=None):
        """Dibuja una etiqueta para identificar la mano y el gesto detectado."""

        try:
            h, w, _ = frame.shape
            cx = int(min(hand_landmarks.landmark[0].x * w, w - 10))
            cy = int(min(hand_landmarks.landmark[0].y * h, h - 10))

            label = f"{hand_type.upper()}"
            if gesture:
                label += f": {gesture}"
                if cmu_notation:
                    config = cmu_notation.get('configuracion', '')
                    mov = cmu_notation.get('movimiento', '')
                    loc = cmu_notation.get('ubicacion', '')
                    label += f" ({config},{mov},{loc})"

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (cx - 5, cy - text_h - 5), (cx + text_w + 5, cy + 5),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

        except Exception as e:
            self.logger.error(f"Error dibujando etiqueta: {e}")

    def _obtener_notacion_cmu(self, gesture):
        """Obtiene la notación CMU para un gesto predefinido."""
        try:
            # Diccionario de señas predefinidas con su notación CMU
            señas_predefinidas = {
                'hoy': (3, 7, 3),
                'en la mañana': (1, 8, 7),
                'tomé': (3, 3, 10),
                'café': (6, 4, 4)
            }

            # Convertir el gesto a minúsculas para coincidencia insensible a mayúsculas
            gesture_lower = gesture.lower()

            # Verificar si el gesto está en las señas predefinidas
            if gesture_lower in señas_predefinidas:
                # Obtener la tupla de notación CMU
                notacion_tupla = señas_predefinidas[gesture_lower]

                # Convertir la tupla a notación de texto usando los labels de CMUDecoder
                notacion = {
                    'configuracion': self.cmu_decoder.configuracion_labels.get(str(notacion_tupla[0]), 'Desconocido'),
                    'movimiento': self.cmu_decoder.movimiento_labels.get(str(notacion_tupla[1]), 'Desconocido'),
                    'ubicacion': self.cmu_decoder.ubicacion_labels.get(str(notacion_tupla[2]), 'Desconocido')
                }

                return notacion

            # Si el gesto no está en las señas predefinidas
            self.logger.warning(f"No se pudo encontrar notación completa para el gesto: {gesture}")
            return None

        except Exception as e:
            self.logger.error(f"Error obteniendo notación CMU: {e}")
            return None

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()