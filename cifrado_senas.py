class CifradoSenas:
    def __init__(self):
        self.modulo = 11
        self.clave_desplazamiento = 2
        self.clave_afin_a = 3
        self.clave_afin_b = 2
        self.clave_vigenere = [1, 7, 3]

    def cifrar(self, notacion, metodo):
        """Cifra una tupla de 3 valores CMU según el método especificado."""
        if not isinstance(notacion, tuple) or len(notacion) != 3:
            raise ValueError("La notación debe ser una tupla de 3 valores")

        if metodo == 'simple':
            return tuple((x + self.clave_desplazamiento) % self.modulo for x in notacion)
        elif metodo == 'afin':
            return tuple((self.clave_afin_a * x + self.clave_afin_b) % self.modulo for x in notacion)
        elif metodo == 'vigenere':
            return tuple((x + self.clave_vigenere[i % len(self.clave_vigenere)]) % self.modulo
                         for i, x in enumerate(notacion))
        else:
            raise ValueError("Método de cifrado desconocido")