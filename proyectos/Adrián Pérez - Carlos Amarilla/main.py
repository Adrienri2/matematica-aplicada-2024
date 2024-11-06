# main.py

import pandas as pd
import time
from preprocessing import preprocess_text
from sentiment_lexicon import calculate_sentiment_scores
from fuzzy_logic import (
    create_membership_functions,
    fuzzy_inference,
    defuzzify,
    get_sentiment_label
)
from benchmark import calculate_benchmarks
import nltk

# Mostrar la versión de NLTK instalada
print("Versión de NLTK:", nltk.__version__)

# Descargar recursos de NLTK necesarios si no están disponibles
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')

def load_and_preprocess():
    """
    Carga el dataset Sentiment140, realiza el preprocesamiento del texto
    y prepara el DataFrame para su posterior análisis.

    Returns:
        df (pd.DataFrame): DataFrame preprocesado con columnas 'text', 'target' y 'clean_text'.
    """
    # Leer el dataset y seleccionar las columnas relevantes
    df = pd.read_csv('sentiment140.csv')
    df = df[['sentence', 'sentiment']]
    
    # Mostrar los valores únicos en la columna 'sentiment'
    print("Valores únicos en 'sentiment':", df['sentiment'].unique())
    
    # Mapear los valores numéricos de 'sentiment' a etiquetas de texto
    df['sentiment'] = df['sentiment'].map({0: 'negative', 1: 'positive', 2: 'neutral'})
    
    # Renombrar columnas para consistencia
    df.rename(columns={'sentence': 'text', 'sentiment': 'target'}, inplace=True)
    
    # Reemplazar valores faltantes en 'text' y asegurar que es de tipo string
    df['text'] = df['text'].fillna('')
    df['text'] = df['text'].astype(str)
    
    # Aplicar preprocesamiento al texto
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    return df

def calculate_scores(df):
    """
    Calcula los puntajes de sentimiento positivos y negativos para cada texto preprocesado.

    Args:
        df (pd.DataFrame): DataFrame con la columna 'clean_text'.

    Returns:
        df (pd.DataFrame): DataFrame con nuevas columnas 'positive_score' y 'negative_score'.
    """
    start_time = time.time()
    # Aplicar la función de cálculo de puntajes a cada texto
    df[['positive_score', 'negative_score']] = df['clean_text'].apply(
        lambda text: pd.Series(calculate_sentiment_scores(text))
    )
    end_time = time.time()
    print(f"Tiempo total para calcular puntajes de sentimiento: {end_time - start_time:.2f} segundos")
    return df

def apply_fuzzy_logic(df):
    """
    Aplica la lógica difusa a los puntajes de sentimiento para obtener el
    puntaje de inferencia y la etiqueta de sentimiento correspondiente.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'positive_score' y 'negative_score'.

    Returns:
        df (pd.DataFrame): DataFrame con nuevas columnas 'sentiment_score',
                           'sentiment_label' y 'execution_time'.
    """
    # Verificar que hay al menos dos clases de sentimiento en el dataset
    if df['target'].nunique() < 2:
        print("El dataset no tiene suficientes clases de sentimiento para aplicar la lógica difusa.")
        return df
    
    # Obtener los valores mínimos y máximos de los puntajes positivos y negativos
    pos_min = df['positive_score'].min()
    pos_max = df['positive_score'].max()
    neg_min = df['negative_score'].min()
    neg_max = df['negative_score'].max()
    
    # Crear las funciones de membresía difusas basadas en los puntajes
    mf = create_membership_functions(pos_min, pos_max, neg_min, neg_max)
    
    # Definir una función interna para procesar cada tweet
    def process_tweet(row):
        """
        Procesa un tweet aplicando la inferencia difusa y calcula el tiempo de ejecución.

        Args:
            row (pd.Series): Fila del DataFrame con 'positive_score' y 'negative_score'.

        Returns:
            pd.Series: Serie con 'sentiment_score', 'sentiment_label' y 'execution_time'.
        """
        start_time = time.time()
        # Aplicar la inferencia difusa a los puntajes
        aggregated, x_op = fuzzy_inference(row['positive_score'], row['negative_score'], mf)
        # Defuzzificar el resultado para obtener el puntaje de sentimiento
        sentiment_score = defuzzify(aggregated, x_op)
        # Obtener la etiqueta de sentimiento basada en el puntaje
        sentiment_label = get_sentiment_label(sentiment_score)
        end_time = time.time()
        # Calcular el tiempo de ejecución para este tweet
        execution_time = end_time - start_time
        return pd.Series([sentiment_score, sentiment_label, execution_time])
    
    # Aplicar la función de procesamiento a cada fila del DataFrame
    df[['sentiment_score', 'sentiment_label', 'execution_time']] = df.apply(process_tweet, axis=1)
    
    return df

def perform_benchmarks(df):
    """
    Calcula y muestra los benchmarks de rendimiento, incluyendo:
    - Total de tweets por categoría de sentimiento.
    - Tiempo promedio de ejecución por categoría.
    - Tiempo promedio total de ejecución.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'sentiment_label' y 'execution_time'.
    """
    calculate_benchmarks(df)

def save_results(df):
    """
    Guarda los resultados del análisis de sentimiento en un archivo CSV.

    Args:
        df (pd.DataFrame): DataFrame con los resultados a guardar.
    """
    # Seleccionar y renombrar las columnas para el archivo de salida
    df_output = df[['text', 'target', 'positive_score', 'negative_score',
                    'sentiment_score', 'sentiment_label', 'execution_time']]
    df_output.columns = ['oracion_original', 'label_original', 'puntaje_positivo',
                         'puntaje_negativo', 'resultado_inferencia', 'sentimiento', 'tiempo_ejecucion']
    # Guardar el DataFrame en un archivo CSV sin índice
    df_output.to_csv('resultado_sentimiento.csv', index=False)
    print("Resultados guardados en 'resultado_sentimiento.csv'.")

def main():
    """
    Función principal que coordina el flujo del análisis de sentimiento:
    - Carga y preprocesa el dataset.
    - Calcula los puntajes de sentimiento.
    - Aplica la lógica difusa para inferir el sentimiento.
    - Realiza benchmarks de rendimiento.
    - Guarda los resultados en un archivo CSV.
    """
    # Cargar y preprocesar el dataset
    df = load_and_preprocess()
    # Verificar que el DataFrame no está vacío
    if df.empty:
        print("El dataframe está vacío. Verifica el archivo CSV y su formato.")
        return
    # Calcular los puntajes de sentimiento
    df = calculate_scores(df)
    # Aplicar la lógica difusa
    df = apply_fuzzy_logic(df)
    # Realizar benchmarks
    perform_benchmarks(df)
    # Guardar los resultados en un archivo CSV
    save_results(df)

# Ejecutar la función principal si el script es ejecutado directamente
if __name__ == '__main__':
    main()
