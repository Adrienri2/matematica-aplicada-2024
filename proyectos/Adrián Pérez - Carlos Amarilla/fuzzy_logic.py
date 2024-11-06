# fuzzy_logic.py

import numpy as np
import skfuzzy as fuzz

def create_membership_functions(pos_min, pos_max, neg_min, neg_max):
    """
    Crea las funciones de membresía difusas para los puntajes positivos y negativos,
    así como para la salida del sistema difuso.

    Args:
        pos_min (float): Valor mínimo de los puntajes positivos.
        pos_max (float): Valor máximo de los puntajes positivos.
        neg_min (float): Valor mínimo de los puntajes negativos.
        neg_max (float): Valor máximo de los puntajes negativos.

    Returns:
        dict: Un diccionario que contiene las funciones de membresía y los universos de discurso.
    """
    # Definir universos de discurso para los puntajes positivos, negativos y la salida
    x_pos = np.linspace(pos_min, pos_max, 100)  # Universo de discurso para puntajes positivos
    x_neg = np.linspace(neg_min, neg_max, 100)  # Universo de discurso para puntajes negativos
    x_op = np.linspace(0, 10, 100)              # Universo de discurso para la salida (0 a 10)

    # Funciones de membresía triangulares para puntajes positivos
    pos_lo = fuzz.trimf(x_pos, [pos_min, pos_min, (pos_min + pos_max)/2])
    pos_md = fuzz.trimf(x_pos, [pos_min, (pos_min + pos_max)/2, pos_max])
    pos_hi = fuzz.trimf(x_pos, [(pos_min + pos_max)/2, pos_max, pos_max])

    # Funciones de membresía triangulares para puntajes negativos
    neg_lo = fuzz.trimf(x_neg, [neg_min, neg_min, (neg_min + neg_max)/2])
    neg_md = fuzz.trimf(x_neg, [neg_min, (neg_min + neg_max)/2, neg_max])
    neg_hi = fuzz.trimf(x_neg, [(neg_min + neg_max)/2, neg_max, neg_max])

    # Funciones de membresía triangulares para la salida (sentimiento)
    op_negative = fuzz.trimf(x_op, [0, 0, 5])    # Sentimiento negativo
    op_neutral = fuzz.trimf(x_op, [0, 5, 10])    # Sentimiento neutral
    op_positive = fuzz.trimf(x_op, [5, 10, 10])  # Sentimiento positivo

    # Almacenar todas las funciones de membresía y universos en un diccionario
    membership_functions = {
        'x_pos': x_pos,
        'x_neg': x_neg,
        'x_op': x_op,
        'pos_lo': pos_lo,
        'pos_md': pos_md,
        'pos_hi': pos_hi,
        'neg_lo': neg_lo,
        'neg_md': neg_md,
        'neg_hi': neg_hi,
        'op_negative': op_negative,
        'op_neutral': op_neutral,
        'op_positive': op_positive
    }

    return membership_functions

def fuzzy_inference(pos_score, neg_score, mf):
    """
    Aplica el sistema de inferencia difusa basado en las reglas definidas
    para calcular el sentimiento a partir de los puntajes positivos y negativos.

    Args:
        pos_score (float): Puntaje positivo del texto.
        neg_score (float): Puntaje negativo del texto.
        mf (dict): Diccionario de funciones de membresía y universos creado por 'create_membership_functions'.

    Returns:
        tuple:
            - aggregated (numpy.array): Función de membresía agregada después de aplicar todas las reglas.
            - x_op (numpy.array): Universo de discurso para la salida, necesario para la defuzzificación.
    """
    # Extraer universos y funciones de membresía del diccionario
    x_pos = mf['x_pos']
    x_neg = mf['x_neg']
    x_op = mf['x_op']
    pos_lo = mf['pos_lo']
    pos_md = mf['pos_md']
    pos_hi = mf['pos_hi']
    neg_lo = mf['neg_lo']
    neg_md = mf['neg_md']
    neg_hi = mf['neg_hi']
    op_negative = mf['op_negative']
    op_neutral = mf['op_neutral']
    op_positive = mf['op_positive']

    # Fuzzificación de los puntajes de entrada
    pos_level_lo = fuzz.interp_membership(x_pos, pos_lo, pos_score)
    pos_level_md = fuzz.interp_membership(x_pos, pos_md, pos_score)
    pos_level_hi = fuzz.interp_membership(x_pos, pos_hi, pos_score)

    neg_level_lo = fuzz.interp_membership(x_neg, neg_lo, neg_score)
    neg_level_md = fuzz.interp_membership(x_neg, neg_md, neg_score)
    neg_level_hi = fuzz.interp_membership(x_neg, neg_hi, neg_score)

    # Aplicación de las nueve reglas difusas

    # Regla 1: Si Positivo es Bajo y Negativo es Bajo entonces Sentimiento es Neutral
    rule1 = np.fmin(pos_level_lo, neg_level_lo)
    sentiment_activation_rule1 = np.fmin(rule1, op_neutral)

    # Regla 2: Si Positivo es Medio y Negativo es Bajo entonces Sentimiento es Positivo
    rule2 = np.fmin(pos_level_md, neg_level_lo)
    sentiment_activation_rule2 = np.fmin(rule2, op_positive)

    # Regla 3: Si Positivo es Alto y Negativo es Bajo entonces Sentimiento es Positivo
    rule3 = np.fmin(pos_level_hi, neg_level_lo)
    sentiment_activation_rule3 = np.fmin(rule3, op_positive)

    # Regla 4: Si Positivo es Bajo y Negativo es Medio entonces Sentimiento es Negativo
    rule4 = np.fmin(pos_level_lo, neg_level_md)
    sentiment_activation_rule4 = np.fmin(rule4, op_negative)

    # Regla 5: Si Positivo es Medio y Negativo es Medio entonces Sentimiento es Neutral
    rule5 = np.fmin(pos_level_md, neg_level_md)
    sentiment_activation_rule5 = np.fmin(rule5, op_neutral)

    # Regla 6: Si Positivo es Alto y Negativo es Medio entonces Sentimiento es Positivo
    rule6 = np.fmin(pos_level_hi, neg_level_md)
    sentiment_activation_rule6 = np.fmin(rule6, op_positive)

    # Regla 7: Si Positivo es Bajo y Negativo es Alto entonces Sentimiento es Negativo
    rule7 = np.fmin(pos_level_lo, neg_level_hi)
    sentiment_activation_rule7 = np.fmin(rule7, op_negative)

    # Regla 8: Si Positivo es Medio y Negativo es Alto entonces Sentimiento es Negativo
    rule8 = np.fmin(pos_level_md, neg_level_hi)
    sentiment_activation_rule8 = np.fmin(rule8, op_negative)

    # Regla 9: Si Positivo es Alto y Negativo es Alto entonces Sentimiento es Neutral
    rule9 = np.fmin(pos_level_hi, neg_level_hi)
    sentiment_activation_rule9 = np.fmin(rule9, op_neutral)

    # Agregación de todas las reglas (unión de todas las salidas activadas)
    aggregated = np.fmax(
        sentiment_activation_rule1,
        np.fmax(
            sentiment_activation_rule2,
            np.fmax(
                sentiment_activation_rule3,
                np.fmax(
                    sentiment_activation_rule4,
                    np.fmax(
                        sentiment_activation_rule5,
                        np.fmax(
                            sentiment_activation_rule6,
                            np.fmax(
                                sentiment_activation_rule7,
                                np.fmax(
                                    sentiment_activation_rule8,
                                    sentiment_activation_rule9
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    # Retorna la función de membresía agregada y el universo de discurso de la salida
    return aggregated, x_op

def defuzzify(aggregated, x_op):
    """
    Defuzzifica la función de membresía agregada para obtener un puntaje numérico de sentimiento.

    Args:
        aggregated (numpy.array): Función de membresía agregada de la salida.
        x_op (numpy.array): Universo de discurso de la salida.

    Returns:
        float: Puntaje numérico de sentimiento obtenido mediante defuzzificación.
    """
    # Defuzzificación utilizando el método del centroide
    sentiment_score = fuzz.defuzz(x_op, aggregated, 'centroid')
    return sentiment_score

def get_sentiment_label(score):
    """
    Asigna una etiqueta de sentimiento ('negative', 'neutral', 'positive') basada en el puntaje numérico.

    Args:
        score (float): Puntaje numérico de sentimiento.

    Returns:
        str: Etiqueta de sentimiento correspondiente.
    """
    if score < 3.3:
        return 'negative'
    elif score > 6.7:
        return 'positive'
    else:
        return 'neutral'
