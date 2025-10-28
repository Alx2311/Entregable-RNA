import nltk # Importa la biblioteca NLTK, utilizada para procesamiento de lenguaje natural (NLP)
from nltk.tokenize import word_tokenize # Importa el método para tokenizar texto en palabras
from nltk.tokenize import TweetTokenizer # Importa un tokenizador especializado para tweets (uso industrial)


# Descargamos el modelo necesario para la tokenización de palabras.
# 'punkt' es el modelo correcto para tokenización, 'punkt_tab' es incorrecto.
nltk.download('punkt') # Descarga el modelo de tokenización de oraciones/palabras

# Definimos el texto que vamos a tokenizar
texto = 'Hola ¿Como esta? hoy aprenderemos porcesamieto de lengauje natural'

# Tokenizamos el texto usando el método word_tokenize (divide el texto en palabras)
tokens = word_tokenize(texto)

print('----tokenizado 01----') # Imprime un separador para identificar la salida

# Volvemos a tokenizar el texto (esto es redundante, pero se muestra como ejemplo)
tokens = word_tokenize(texto)
print(tokens) # Imprime la lista de tokens generados por word_tokenize

print('----tokenizado 02----') # Otro separador para la siguiente salida

# Creamos una instancia del tokenizador especializado para tweets
tokenizador = TweetTokenizer()

# Tokenizamos el texto usando TweetTokenizer (puede manejar mejor emojis, hashtags, etc.)
tokens = tokenizador.tokenize(texto)

print(tokens) # Imprime la lista de tokens generados por TweetTokenizer




