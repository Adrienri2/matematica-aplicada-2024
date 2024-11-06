# benchmark.py

def calculate_benchmarks(df):
    """
    Calcula y muestra estadísticas sobre el análisis de sentimiento realizado,
    incluyendo totales y tiempos promedio por categoría de sentimiento.

    Estadísticas calculadas:
    - Total de tweets por categoría de sentimiento ('positive', 'negative', 'neutral').
    - Tiempo promedio de ejecución para tweets en cada categoría.
    - Tiempo promedio total de ejecución.

    Args:
        df (pd.DataFrame): DataFrame que contiene los resultados del análisis,
                           incluyendo las columnas 'sentiment_label' y 'execution_time'.
    """
    # Calcular el total de tweets por categoría de sentimiento
    total_positive = len(df[df['sentiment_label'] == 'positive'])
    total_negative = len(df[df['sentiment_label'] == 'negative'])
    total_neutral = len(df[df['sentiment_label'] == 'neutral'])

    print(f"Total de tweets positivos: {total_positive}")
    print(f"Total de tweets negativos: {total_negative}")
    print(f"Total de tweets neutrales: {total_neutral}")

    # Calcular el tiempo promedio de ejecución por categoría
    avg_time_positive = df[df['sentiment_label'] == 'positive']['execution_time'].mean()
    avg_time_negative = df[df['sentiment_label'] == 'negative']['execution_time'].mean()
    avg_time_neutral = df[df['sentiment_label'] == 'neutral']['execution_time'].mean()

    print(f"Tiempo promedio de ejecución para tweets positivos: {avg_time_positive:.6f} segundos")
    print(f"Tiempo promedio de ejecución para tweets negativos: {avg_time_negative:.6f} segundos")
    print(f"Tiempo promedio de ejecución para tweets neutrales: {avg_time_neutral:.6f} segundos")

    # Calcular el tiempo promedio total de ejecución
    total_avg_time = df['execution_time'].mean()
    print(f"Tiempo promedio total de ejecución: {total_avg_time:.6f} segundos")
