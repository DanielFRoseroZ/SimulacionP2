import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

#Definir la malla discretizada obtenida en la primera entrega para la velocidad en X
matriz_vel_x = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
                         [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
                         [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
                         [1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [1, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

#Definir la malla discretizada obtenida en la primera entrega para la velocidad en Y
matriz_vel_y = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
                         [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
                         [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
                         [0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

#Funcion que le asigna un id a cada punto a evaluar de la malla generada en la primera entrega
def set_var_id(arr):
    var = -1
    #Verificar el tamaño de la matriz
    y,x = arr.shape

    #Recorrer la matriz y asignar un id a cada punto
    for i in range(y):
        for j in range(x):
            if arr[i][j] == -1:
                arr[i][j] = var
                var += -1
    return arr

#Asignar un id a cada punto de la malla de velocidades en X y en Y
set_var_id(matriz_vel_x)
set_var_id(matriz_vel_y)

#Funcion que verifica si un punto esta dentro de la malla o si es un valor de frontera
def check(i, j, arr):
    if i < 0 or j < 0:
        return "Desbordamiento"
    
    value = arr[i][j]
    if value == 0 or value == 1:
        return "Frontera"

    #Si no es frontera ni desbordamiento, entonces es un punto dentro de la malla
    return abs(value) - 1

#Definir operaciones de las series de taylor de orden superior centradas a aplicar
operaciones = [(-60/12, 0,0), (24/12, 0,-1), (-2/12, 0,-2), (24/12, -1,0), (-2/12, -2,0), (8/12, 0,1), (8/12, 1,0)]

#Funcion que aplica las operaciones de las series de taylor de orden superior centradas a cada punto de la malla
def apply_operator(arr, operaciones, i, j, var, coeficiente, a):
    #Recorrer las operaciones
    for operacion in operaciones:
        #Verificar si el punto esta dentro de la malla o si es un valor de frontera y aplicar la operacion
        val = check(i + operacion[1], j + operacion[2], arr)
        match val:
            case "Desbordamiento":
                continue
            case "Frontera":
                coeficiente[var] += -1 * operacion[0] * arr[i + operacion[1], j + operacion[2]]
            case _:
                a[var][val] = operacion[0]

#Funcion que crea la matriz A y matriz de coeficientes
def create_matriz_A(arr, operacion):
    #Inicializa la matriz A con ceros
    a = np.zeros((69,69))
    #Inicializa la matriz de coeficientes con ceros
    terminos_independientes = np.zeros(69)
    var = 0
    len_y, len_x = arr.shape
    for i in range(1, len_y):
        for j in range(1, len_x):
            if arr[i][j] == 0:
                continue
            apply_operator(arr, operacion, i, j, var, terminos_independientes, a)
            var += 1
    return a, terminos_independientes

#Crear matriz A y matriz de coeficientes para las velocidades en X y en Y
matriz_A_x, vector_b_x = create_matriz_A(matriz_vel_x, operaciones)
matriz_A_y, vector_b_y = create_matriz_A(matriz_vel_y, operaciones)

#print(matriz_A_x)
#print(vector_b_x)

#Funcion que calcula la norma de la diferencia entre dos vectores
def diferencia_vectores(x, y):
    return np.linalg.norm(x - y)



'''
IMPLEMENTACIÓN DEL MÉTODO DE JACOBI CON SOBRERELAJACIÓN
'''

def jacobi_sr(matriz_A, vector_b, n, x=None, omega=1.0, e=0.0001):

    #inicializa el vector x con ceros si no se especifica
    if x is None:
        x = np.zeros_like(vector_b)

    D = np.diag(np.diag(matriz_A)) # Matriz diagonal de A
    L = np.tril(matriz_A, -1) # Matriz triangular inferior de A
    U = np.triu(matriz_A, 1) # Matriz triangular superior de A

    L_plus_U = L + U
    
    for i in range(n):
        x0 = x.copy()
        x = (1.0 - omega) * x0 + omega * np.linalg.inv(D).dot(vector_b - (L_plus_U).dot(x0))

        if(diferencia_vectores(x, x0) < e):
            print("Converge en la iteración: ", i+1)
            break
    #print("Solución: ", x)
    return x

#resultado_jacobi = jacobi_sr(matriz_A_x, vector_b_x, 1000)



'''
IMPLEMENTACIÓN DEL MÉTODO DE GRADIENTE CONJUGADO
'''
def grad_conjugate(matriz_A, vector_b, x=None, e=0.0001, n=1000):

    #inicializa el vector x con ceros si no se especifica
    if x is None:
        x = np.zeros_like(vector_b)
        print(x)
    
    r = vector_b - matriz_A.dot(x) # Se define el residuo como la diferencia entre el vector b y el producto punto de la matriz A y el vector x
    p = r.copy() # Se define la dirección de búsqueda como el residuo

    # Se itera hasta que el residuo sea menor a la tolerancia o se llegue al número máximo de iteraciones
    for i in range(n):
        Ap = matriz_A.dot(p)
        alpha = np.dot(p, r) / np.dot(p, Ap)
        x = x + alpha * p
        r = vector_b - matriz_A.dot(x)
        
        # Se verifica si el residuo es menor a la tolerancia
        if np.sqrt(np.sum((r**2))) < e:
            print("Converge en la iteración: ", i)
            break
        else:
            beta = -np.dot(r, Ap) / np.dot(p, Ap)
            p = r + beta * p      
    return x
    

resultado_jacobi = jacobi_sr(matriz_A_x, vector_b_x, 1000)
resultado_gradiente = grad_conjugate(matriz_A_x, vector_b_x)

print(resultado_jacobi)
print(resultado_gradiente)

print(diferencia_vectores(resultado_jacobi, resultado_gradiente))



'''
IMPLEMENTACION DEL METODO DE NEWTON-RAPHSON
'''