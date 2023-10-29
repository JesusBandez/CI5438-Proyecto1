"""Entrena al modelo"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
class Trainer:
    def __init__(self) -> None:
        self._hipotesis = None

    def gradient_descent(self, X:np.ndarray, Y:np.ndarray, 
            iterations:int, alpha:float, epsilon:float, 
            plot_error:bool=False):
        "Genera una hipotesis con los vectores X e Y"
            
        if np.shape(X)[0] != np.shape(Y)[0]:
            raise ValueError("Los vectores X y Y no tienen la misma longitud")
        
        # Agregar el termino Bias
        bias_term = np.ones(len(X))
        X = np.column_stack((bias_term, X))
        
        m = np.shape(X)[0]
        W = [1 for _ in range(len(X[0]))] #TODO: (Deberiamos usar random)[random.random() for _ in range(len(X[0]))]
        iteration_error = []
        for _ in range(iterations):
            
            P = np.matmul(X,W)
            Xt = X.transpose()
            MSE = np.matmul(Xt,(P-Y))*(2/m)
            iteration_error.append(max(abs(MSE)))
            W = W - (MSE*alpha)
            if max(abs(MSE))<epsilon:
                break

        self._hipotesis = W
        if plot_error:
            
            plt.plot(list(range(0,len(iteration_error))), iteration_error)
            plt.xlabel('Iteracion') 
            plt.ylabel('Error en la iteracion') 
            plt.title('Gráfico') 
            plt.show()
        return W

    def predict(self, data:np.ndarray):
        "Recibe un arreglo de datos para predecir su precio"
        if self._hipotesis is None:
            raise Exception("Aun no se ha entrenado el modelo")
        if len(self._hipotesis)-1 != len(data):
            raise Exception("Lo datos tienen dimensiones incorrectas")
        
        unbiassed_result = sum([W*D for W, D in zip(self._hipotesis[1:], data)])
        return unbiassed_result + self._hipotesis[0]


if __name__ == "__main__":
    # Importar conjunto limpio
    df = pd.read_csv("CleanedCarDekho.csv")
    Y = np.array(df["Price"])
    X = np.array(df.drop("Price",axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7)
    trainer = Trainer()
    hipotesis = trainer.gradient_descent(X, Y, 10000, 0.1, 0.001, False)
    
    test = zip([int(trainer.predict(test)) for test in X_test], [real for real in y_test])
    for predicted, real in test:
        print(f"prediccion: {predicted}, real: {real}")


    



    
    # Convertirlo en los arreglos X e Y
    # Ejecutar el entrenamiento
    # Probar la hipotesis conseguida


    #     # Funcion prueba
    # def f(x1,x2):
    #     w0=2
    #     w1=3
    #     w2=5
    #     return (w1*x1+w2*x2+w0)

    # # Generar conjunto de pruebas
    # X_=[]
    # Y_=[]
    # from random import gauss
    # for x in range(1000):
    #     x1=gauss(0,1)
    #     x2=gauss(0,1)
    #     X_.append([1,x1,x2])
    #     Y_.append(f(x1,x2))

    # X = np.array(X_)
    # Y = np.array(Y_)
    # trainer = Trainer()
    # print(trainer.gradient_descent(X, Y, 100, 0.01, 0.001, True))