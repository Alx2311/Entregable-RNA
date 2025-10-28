import spacy
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 

#

nlp=spacy.load('es_core_news_sm') 

def limpiar_tokenizar(texto):
    doc=nlp(texto.lower()) 
    #tokens= [token.lemma_ for token in doc if not token.is_punct and not ]
    tokens= [token.lemma_ for token in doc if not token.is_punct] 
    return " ".join(tokens)  

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
etiquetas=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]
textos_limpios=[limpiar_tokenizar(t) for t in textos]
x_train, x_test, y_train, y_test=train_test_split(textos_limpios, etiquetas, test_size=0.2, random_state=42)
vectorizer=TfidfVectorizer()
x_train_vectorizado=vectorizer.fit_transform(x_train) 
x_test_vectorizado=vectorizer.transform(x_test) 

#print(x_train_vectorizado)

#-----Implementación de nuestro modelo de machine learning
model=LogisticRegression()
model.fit(x_train_vectorizado, y_train) 

nuevos =[ 
    "La compra fue excelente y rápida",
    "Mala calidad del producto, no funciona",
    "No volvería a comprar jamás",
    "Estoy muy satisfecho con el servicio"
    ]

nuevos_limpios=[limpiar_tokenizar(t) for t in nuevos] 
nuevos_vectorizados=vectorizer.transform(nuevos_limpios) 
print(model.predict(nuevos_vectorizados)) 