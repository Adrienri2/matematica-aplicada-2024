# sentiment_lexicon.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, sentiwordnet as swn
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Inicializar el lematizador de WordNet
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """
    Convierte las etiquetas POS de Treebank a las etiquetas POS de WordNet.

    Args:
        treebank_tag (str): Etiqueta POS de Treebank (por ejemplo, 'NN', 'VB').

    Returns:
        str or None: Etiqueta POS correspondiente en WordNet ('n', 'v', 'a', 'r') o None si no coincide.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ  # Adjetivo
    elif treebank_tag.startswith('V'):
        return wordnet.VERB  # Verbo
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN  # Sustantivo
    elif treebank_tag.startswith('R'):
        return wordnet.ADV  # Adverbio
    else:
        return None  # Si no es una categoría relevante

def calculate_sentiment_scores(text):
    """
    Calcula los puntajes de sentimiento positivos y negativos de un texto dado,
    utilizando SentiWordNet y desambiguación de sentido con el algoritmo de Lesk.

    Args:
        text (str): Texto preprocesado para el cual se calcularán los puntajes de sentimiento.

    Returns:
        tuple: Una tupla (pos_score, neg_score) con los puntajes positivos y negativos acumulados.
    """
    # Tokenizar el texto en palabras
    tokens = word_tokenize(text)
    # Etiquetar cada token con su categoría gramatical (POS tagging)
    pos_tags = pos_tag(tokens)
    # Inicializar los puntajes acumulados
    pos_score = 0.0
    neg_score = 0.0

    # Iterar sobre cada palabra y su etiqueta POS
    for i, (word, tag) in enumerate(pos_tags):
        # Obtener la etiqueta POS de WordNet correspondiente
        wn_tag = get_wordnet_pos(tag)
        if wn_tag not in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV, wordnet.VERB):
            continue  # Si no es una categoría relevante, pasar al siguiente token

        # Lematizar la palabra con la etiqueta POS de WordNet
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)

        # Definir el contexto para la desambiguación (ventana de 10 palabras)
        context = tokens[max(0, i - 5):i + 5]

        # Realizar desambiguación de sentido usando el algoritmo de Lesk
        synset = lesk(context, word, pos=wn_tag)

        if synset is not None:
            try:
                # Obtener los puntajes de sentimiento del synset desde SentiWordNet
                swn_synset = swn.senti_synset(synset.name())
                # Acumular los puntajes positivos y negativos
                pos_score += swn_synset.pos_score()
                neg_score += swn_synset.neg_score()
            except:
                # Si ocurre un error al obtener los puntajes (por ejemplo, el synset no está en SentiWordNet), continuar
                continue

    return pos_score, neg_score
