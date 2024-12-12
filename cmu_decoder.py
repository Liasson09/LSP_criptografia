from cifrado_senas import CifradoSenas

class CMUDecoder:
    def __init__(self, cifrador=None):
        self.cifrador = CifradoSenas()

        self.configuracion_labels = {
            '0': 'Dedo pulgar, índice y meñique levantado y el resto de los dedos doblados.',
            '1': 'Mano completamente abierta.',
            '2': 'Dedo índice levantado y el resto de los dedos doblados.',
            '3': 'Mano cerrada (puño) con el pulgar y meñique hacia afuera.',
            '4': 'Mano en forma de “L” (dedo índice y pulgar en ángulo recto).',
            '5': 'Mano en forma de “V” (dedos índices y medio levantados, el resto doblado).',
            '6': 'Mano cerrada (puño)',
            '7': 'Mano en forma de "O" (dedos tocándose formando un círculo).',
            '8': 'Mano en forma de "C" (dedos curvados formando un semicírculo).',
            '9': 'Dedo índice y medio juntos apuntando hacia adelante, el resto de los dedos doblados.',
            '10': 'Mano en forma de garra (dedos curvados hacia adentro, simulando una garra).'
        }

        self.movimiento_labels = {
            '0': 'Movimiento en línea recta hacia un lado a la izquierda.',
            '1': 'Sin movimiento, la mano se mantiene en una posición estática.',
            '2': 'Movimiento hacia adelante.',
            '3': 'Movimiento hacia atrás.',
            '4': 'Movimiento circular en el sentido de las agujas del reloj.',
            '5': 'Movimiento en línea recta hacia un lado a la derecha.',
            '6': 'Movimiento hacia arriba.',
            '7': 'Movimiento hacia abajo.',
            '8': 'Movimiento diagonal hacia arriba a la derecha.',
            '9': 'Movimiento diagonal hacia abajo a la izquierda.',
            '10': 'Movimiento en espiral (amplio o cerrado).'
        }

        self.ubicacion_labels = {
            '0': 'Lado izquierdo de la cabeza.',
            '1': 'En la parte superior de la cabeza.',
            '2': 'Frente a la cara, a la altura de los ojos.',
            '3': 'A la altura del pecho.',
            '4': 'Abajo, cerca de la cintura.',
            '5': 'A un costado del cuerpo, cerca del hombro.',
            '6': 'Frente al cuello.',
            '7': 'A la altura del abdomen.',
            '8': 'A un costado del cuerpo, más cerca de la cadera.',
            '9': 'Lado derecho de la cabeza.',
            '10': 'Frente a la cara, a la altura de la boca.'
        }



    def descifrar(self, valor_cifrado, indice, metodo, cifrador):
        """Descifra un solo dígito CMU."""
        categorias = ['configuracion', 'movimiento', 'ubicacion']
        categoria = categorias[indice]
        modulo = cifrador.modulo

        if metodo == 'simple':
            valor = (valor_cifrado - cifrador.clave_desplazamiento) % modulo
        elif metodo == 'afin':
            inverso_a = pow(cifrador.clave_afin_a, -1, modulo)
            valor = (inverso_a * (valor_cifrado - cifrador.clave_afin_b)) % modulo
        elif metodo == 'vigenere':
            valor = (valor_cifrado - cifrador.clave_vigenere[indice % len(cifrador.clave_vigenere)]) % modulo
        else:
            raise ValueError("Método desconocido")

        return self._obtener_descripcion(valor, categoria)

    def _obtener_descripcion(self, valor, categoria):
        """Obtiene la descripción de una categoría según el valor."""
        if categoria == 'configuracion':
            return self.configuracion_labels.get(str(valor), "Desconocido")
        elif categoria == 'movimiento':
            return self.movimiento_labels.get(str(valor), "Desconocido")
        elif categoria == 'ubicacion':
            return self.ubicacion_labels.get(str(valor), "Desconocido")
        else:
            return "Categoría desconocida"