# Análisis de Sentimiento con Reglas Difusas

Este proyecto implementa un sistema de análisis de sentimiento basado en reglas difusas, siguiendo el artículo de Vashishtha y Susan (2019). El programa procesa el dataset Sentiment140, calcula puntajes de sentimiento utilizando SentiWordNet y aplica lógica difusa para clasificar los tweets como positivos, negativos o neutrales.

1. **Instala `virtualenv`**:

   Abre una terminal en el directorio del proyecto y ejecuta:

   ```bash
   pip install virtualenv
   ```

2. **Crea un entorno virtual**:

   Crea el entorno virtual "entorno1":

   ```bash
   virtualenv entorno1
   ```

3. **Activa el entorno virtual**:

   En Windows, ejecuta:

   ```bash
   .\entorno1\Scripts\activate
   ```

4. **Instala las dependencias en el entorno virtual**:

   Con el entorno activado, instala las dependencias usando:

   ```bash
   pip install -r requirements.txt
   ```

5. **Verifica el dataset**:

   Asegúrate de que el archivo del dataset `sentiment140.csv` esté en la raíz del proyecto.

6. **Ejecuta el programa**:

   Ejecuta el programa con:

   ```bash
   python main.py
   ```

7. **Resultados**:

   El resultado del análisis de sentimiento se guardará en `resultado_sentimiento.csv` en la raíz del proyecto.