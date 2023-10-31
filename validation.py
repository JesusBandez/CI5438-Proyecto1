import numpy as np
from random import gauss, random
import matplotlib.pyplot as plt 

w0 = 0.6294111443016626 #random()
w1 = 0.8049650781942079 #random()
w2 = 0.8577551469796346 #random()


def f(x1,x2):
  return (w1*x1+w2*x2+w0)

def descendiente_gradiente(iteraciones,alpha,epsilon,num):
	X_=[]
	Y_=[]

	for x in range(1000):
		x1=gauss(0,1)
		x2=gauss(0,1)
		X_.append([1,x1,x2])
		Y_.append(f(x1,x2))

	X = np.array(X_)
	dimensions = np.shape(X)
	m = dimensions[0]
	Y = np.array(Y_)
	W = np.array([1,1,1])

	error_iteracion=[]
	for it in range(iteraciones):
		P = np.matmul(X,W)
		Xt = X.transpose()
		MSE = np.matmul(Xt,(P-Y))*(2/m)
		error_iteracion.append(max(abs(MSE)))
		W = W - (MSE*alpha)
		if max(abs(MSE))<epsilon:
			break

	SSres = 0
	SStot = 0 
	average = sum(Y)/len(Y)
	for predicted, real in zip(P,Y):
	    SSres += (real - predicted)**2
	    SStot += (real - average)**2
	coeff = 1-(SSres/SStot)

	plt.plot(list(range(0,len(error_iteracion))), error_iteracion) 
	plt.xlabel('Iteración') 
	plt.ylabel('Error en la iteración') 
	plt.title("Curva de entrenamiento \n it={} alpha={} epsilon={}".format(iteraciones,alpha,epsilon))
	plt.savefig("plots/plot_{}.jpg".format(num), dpi=300)
	plt.clf()

	return(coeff)

print("W0",w0,"W1",w1,"W2",w2)
print(descendiente_gradiente(100,0.01,0.001,1))

w0 = 7890
w1 = 1456
w2 = 6786

print("W0",w0,"W1",w1,"W2",w2)
print(descendiente_gradiente(1000,0.01,0.0001,2))

w0 = 14578
w1 = 58962
w2 = 95653

print("W0",w0,"W1",w1,"W2",w2)
print(descendiente_gradiente(1000,0.01,0.0001,3))