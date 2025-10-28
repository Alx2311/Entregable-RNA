import nltk # Importa la biblioteca NLTK para procesamiento de lenguaje natural
from nltk.tokenize import TweetTokenizer # Importa un tokenizador especializado para tweets
# Corpus: colección grande y estructurada de texto
# Stopword: palabras que no proporcionan significado importante (como "el", "la", "de", etc.)
from nltk.corpus import stopwords # Importa el módulo de stopwords de NLTK

# Descargamos el corpus de stopwords (palabras vacías)
nltk.download('stopwords')

# Definimos el texto a procesar
texto ='Hola ¿Como esta? hoy aprenderemos porcesamieto de lengauje natural'

#--tokenización--
# Creamos una instancia del tokenizador para tweets
tokenizador=TweetTokenizer()
# Tokenizamos el texto convirtiéndolo primero a minúsculas
tokens=tokenizador.tokenize(texto.lower())

print('--texto tokenizado--') # Imprime un separador para identificar la salida
print(tokens) # Imprime la lista de tokens generados

# Limpieza de stop_words
# Obtenemos la lista de stopwords en español
stop_words=stopwords.words('spanish')

print('--corpus de stop_words--') # Imprime un separador
print(stop_words) # Imprime la lista de stopwords

print('---limpeza de texto---') # Imprime un separador

# Eliminamos tokens que no sean palabras alfabéticas y que estén en la lista de stopwords
texto_limpio=[t for t in tokens if t.isalpha() and t not in stop_words]

print(texto_limpio) # Imprime la lista de tokens limpios (sin stopwords)