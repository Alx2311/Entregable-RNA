import spacy
from sklearn.feature_extraction.text import TfidfVectorizer # Convierte texto en vectores numéricos ponderados usando TF-IDF
from sklearn.naive_bayes import MultinomialNB # Modelo clásico utilizado para etiquetar texto (clasificación)
from sklearn.model_selection import train_test_split # Para dividir los datos en entrenamiento y prueba

#
                                                                                                      
nlp=spacy.load('es_core_news_sm') # Carga el modelo de lenguaje en español de spaCy

def limpiar_tokenizar(texto):
    doc=nlp(texto.lower()) # Convierte el texto a minúsculas y lo procesa con spaCy
    #tokens= [token.lemma_ for token in doc if not token.is_punct and not ]
    tokens= [token.lemma_ for token in doc if not token.is_punct] # Extrae los lemas de los tokens, omitiendo signos de puntuación
    return " ".join(tokens)  # Une los lemas en una sola cadena separada por espacios

# Texto en lenguaje natural (sin normalización previa)
textos=[
    "Me encanta este producto",
    "Es horrible, no lo recomiendo",
    "Excelente atención al cliente",
    "La experiencia fue muy mala",
    "Estoy feliz con la compra",
    "No me gustó para nada",
    "Fue una buena compra",
    "Una pésima decisión",
    "El servicio fue excelente",
    "No volveré jamás",
    "No me gustó nada el servicio",
    "Jamás volvería a comprar aquí",
    "Nada de lo prometido fue cumplido",
    "Muy lento y poco profesional",
    "Estoy muy satisfecho con el servicio",
    "Fue una experiencia muy agradable"
]
# Etiquetas normalizadas (0 = negativo, 1 = positivo)
etiquetas=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]
# Limpieza de textos (tokenización y lematización)
textos_limpios=[limpiar_tokenizar(t) for t in textos]
# Dividimos los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test=train_test_split(textos_limpios, etiquetas, test_size=0.2, random_state=42)
# Vectorización de texto: convierte los textos en vectores numéricos
vectorizer=TfidfVectorizer()
x_train_vectorizado=vectorizer.fit_transform(x_train) # Ajusta el vectorizador y transforma los textos de entrenamiento
x_test_vectorizado=vectorizer.transform(x_test) # Transforma los textos de prueba usando el mismo vectorizador

#print(x_train_vectorizado)



#-----Implementación de nuestro modelo de machine learning
# Declaramos nuestro modelo de clasificación binaria de texto
model=MultinomialNB()
model.fit(x_train_vectorizado, y_train) # Entrenamos el modelo con los datos vectorizados

# Declaramos nuevas muestras para la predicción
nuevos =[ 
    "La compra fue excelente y rápida",
    "Mala calidad del producto, no funciona",
    "No volvería a comprar jamás",
    "Estoy muy satisfecho con el servicio"
    ]

nuevos_limpios=[limpiar_tokenizar(t) for t in nuevos] # Limpiamos y lematizamos los nuevos textos
nuevos_vectorizados=vectorizer.transform(nuevos_limpios) # Vectorizamos los nuevos textos
print(model.predict(nuevos_vectorizados)) # Imprimimos las predicciones del modelo para los nuevos textos