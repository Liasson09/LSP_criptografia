from cifrado_senas import CifradoSenas
from cmu_decoder import CMUDecoder

class TraductorSenas:
    def __init__(self):
        self.cifrador = CifradoSenas()
        self.cmu_decoder = CMUDecoder()

    def convertir_notacion_a_numeros(self, notacion_dict):
        """Convierte la notación CMU de formato diccionario a tupla numérica."""
        try:
            config_map = {v: int(k) for k, v in self.cmu_decoder.configuracion_labels.items()}
            mov_map = {v: int(k) for k, v in self.cmu_decoder.movimiento_labels.items()}
            loc_map = {v: int(k) for k, v in self.cmu_decoder.ubicacion_labels.items()}

            config_val = config_map.get(notacion_dict['configuracion'], 0)
            mov_val = mov_map.get(notacion_dict['movimiento'], 0)
            loc_val = loc_map.get(notacion_dict['ubicacion'], 0)

            return (config_val, mov_val, loc_val)

        except Exception as e:
            print(f"Error convirtiendo notación a números: {e}")
            return (0, 0, 0)

    def cifrar_notacion_cmu(self, notacion_dict, metodo='simple'):
        """Cifra una notación CMU usando el método especificado."""
        try:
            notacion_numerica = self.convertir_notacion_a_numeros(notacion_dict)

            if metodo == 'simple':
                return self.cifrador.cifrado_simple(notacion_numerica)
            elif metodo == 'afin':
                return self.cifrador.cifrado_afin(notacion_numerica)
            elif metodo == 'vigenere':
                return self.cifrador.cifrado_vigenere(notacion_numerica)
            else:
                raise ValueError(f"Método de cifrado '{metodo}' no válido")

        except Exception as e:
            print(f"Error cifrando notación CMU: {e}")
            return None