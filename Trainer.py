"""Entrena al modelo"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import random
class Trainer:
    def __init__(self) -> None:
        self._hipotesis = None
        self._W = []

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
        #W = [1 for _ in range(len(X[0]))] #TODO: (Deberiamos usar random)[random.random() for _ in range(len(X[0]))]
        W = np.append(1,self._W)
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
    # Cargar el dataframe y dividirlo
    df = pd.read_csv("CleanedCarDekho.csv")
    Y = np.array(df["Price"])
    X = np.array(df.drop("Price",axis=1))

    # Cargar los conjuntos de pruebas
    file_names = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    if not all([os.path.exists(os.path.join("train_data", file)) for file in file_names]):
        # Si no existen, se generan nuevos (Necesitas sklearn instalado)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)        
        np.savetxt(os.path.join("train_data", "X_train.csv"), X_train, delimiter=",")
        np.savetxt(os.path.join("train_data", "X_test.csv"), X_test, delimiter=",")
        np.savetxt(os.path.join("train_data", "y_train.csv"), y_train, delimiter=",")
        np.savetxt(os.path.join("train_data", "y_test.csv"), y_test, delimiter=",")
    else:
        X_train = np.genfromtxt(os.path.join("train_data","X_train.csv") , delimiter=",")
        X_test = np.genfromtxt(os.path.join("train_data", "X_test.csv"), delimiter=",")
        y_train = np.genfromtxt(os.path.join("train_data", "y_train.csv"), delimiter=",")
        y_test = np.genfromtxt(os.path.join("train_data", "y_test.csv"), delimiter=",")

    
    # Conseguir la hipotesis
    trainer = Trainer()
    trainer._W = X[0]
    hipotesis = trainer.gradient_descent(X_train, y_train, 100000, 0.01, 0.001, False)
    
    # Probar la hipotesis
    tests_results = list(zip([int(trainer.predict(test)) for test in X_test], [real for real in y_test]))
    for predicted, real in tests_results:
        print(f"real: {real}, prediccion: {predicted}")

    # Calculo del error medio relativo
    
    relative_errors = [abs(predicted-real)/real for predicted, real in tests_results]
    ERM = sum(relative_errors)/len(tests_results)
    print(f"Error relativo promedio: {ERM}")
    print(f"Mayor error: {round(max(relative_errors), 4)}")
    print(f"Menor error: {round(min(relative_errors), 4)}")

    #Mean Absolute Error (MAE)
    count = 0
    n = len(tests_results)
    for predicted, real in tests_results:
        count += abs(predicted-real)
    print("MAE: ",count/n)

    #Mean Squared Error (MSE)
    count1 = 0
    for predicted, real in tests_results:
        count1 += (real-predicted)**2
    print("MSE: ",(1/n)*count1)

    #Root Mean Square Error (RMSE)
    count2 = 0
    for predicted, real in tests_results:
        count2 += (predicted - real)**2
    print("RMSD: ", math.sqrt(count2/n))

    #Coefficient of determination
    SSres = 0
    SStot = 0 
    average = sum(y_test)/len(y_test)
    for predicted, real in tests_results:
        SSres += (real - predicted)**2
        SStot += (real - average)**2
    print("Coeff: ",1-(SSres/SStot))
    #R² (percentage of variance explained by model)
