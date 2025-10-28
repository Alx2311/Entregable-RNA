import spacy # Importamos la biblioteca spaCy para procesamiento de lenguaje natural (NLP)
# spacy.cli.download('es_core_news_sm') # Línea para descargar el modelo en español (descomentala si no tienes el modelo)
nlp = spacy.load('es_core_news_sm') # Cargamos el modelo de spaCy para español

# Definimos el texto a analizar
texto = 'El jugador de futbol Alberto Medina publico que entrenara para un equipo en puno'

# Procesamos el texto con el modelo cargado
doc = nlp(texto)

print('--Seleccion de entidades nombradas--') # Imprime un separador para la salida de entidades nombradas

# Iteramos sobre las entidades nombradas detectadas en el texto
for t in doc.ents: # doc.ents contiene las entidades nombradas encontradas
    print(t) # Imprime cada entidad nombrada

print('---Etiquetado Gramatical---') # Imprime un separador para la salida de etiquetas gramaticales

# Iteramos sobre cada token del texto para mostrar su etiqueta gramatical
for t in doc:
    print(t.text, t.pos_) # Imprime el texto del token y su categoría gramatical

print('----Lematizacion texto----') # Imprime un separador para la salida de lematización

# Iteramos sobre cada token para mostrar su lema (forma base de la palabra)
for t in doc:
    print(t.text, t.lemma_) # Imprime el texto original y su lema