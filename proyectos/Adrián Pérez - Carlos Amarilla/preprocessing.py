# preprocessing.py

import re

def preprocess_text(text):
    """
    Preprocesa el texto eliminando y transformando elementos para preparar los datos
    para el análisis de sentimiento.

    Pasos realizados:
    - Elimina URLs y menciones de usuarios (@usuario).
    - Expande contracciones comunes (por ejemplo, "can't" a "cannot").
    - Elimina el símbolo '#' manteniendo la palabra del hashtag.
    - Elimina caracteres especiales y números, manteniendo solo letras y espacios.
    - Convierte el texto a minúsculas.
    - Elimina espacios adicionales al inicio y al final del texto.

    Args:
        text (str): Texto original que se desea preprocesar.

    Returns:
        str: Texto preprocesado y limpio.
    """
    # Eliminar URLs y menciones (@usuario)
    text = re.sub(r'http\S+|www\.\S+|@\w+', '', text)

    # Expansión de contracciones comunes
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not"  # Abarca contracciones como "didn't", "isn't", etc.
    }
    for contraction, full_form in contractions.items():
        text = text.replace(contraction, full_form)

    # Eliminar el símbolo '#' pero mantener la palabra del hashtag
    text = re.sub(r'#', '', text)

    # Eliminar caracteres especiales y números, manteniendo solo letras y espacios
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Convertir todo el texto a minúsculas
    text = text.lower()

    # Eliminar espacios en blanco adicionales al inicio y al final
    return text.strip()
