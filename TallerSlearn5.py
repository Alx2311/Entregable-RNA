#-----Reg. Logistica
from sklearn.linear_model import LogisticRegression #Importamos el modelo para clasificacion binaria
x=[[-5],[-4],[-3],[-2],[-1],[1],[2],[3],[4],[5]] #5 datos con 1 caracteristica (5*1)
y=[0,0,0,0,0,1,1,1,1,1,] #(1 * 10)--> Etiquetas en binario: 0 - y1 +
model=LogisticRegression() #Declaramos el modelo
model.fit(x,y)  #Entrenamos con suestrso datos
print(model.predict([[20]]))  #Realizamos una prediccion