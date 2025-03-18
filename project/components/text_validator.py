"""
Clase para validar campos de tipo texto.
"""
class TextValidator():
    def __init__(self):
        pass
    
    # Para validar un numero
    @staticmethod
    def validate_number(text):
        try:
            number=float(text)
            return number
        except ValueError:
            return None
        
    # Para validar una tupla de 2 elementos
    @staticmethod
    def validate_tuple2(text):
        split=text.split(",")
        if len(split) == 2:
            val1=TextValidator.validate_number(text=split[0])
            val2=TextValidator.validate_number(text=split[1])
            if val1 is not None and val2 is not None:
                return (int(val1),int(val2))
            else:
                return None
        else:
            return None