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

def norm(vector1, vector2):
    '''
    Calcula la diferencia de valores entre dos vectores
    '''
    suma = 0.0
    for i in range(len(vector1)):
        suma += vector1[i] - vector2[i]

    return abs(suma)

    
def jacobi(A, b, n, x=None, omega=0, e=0.0001, prnt=True):
    '''
    Resuelve el sistema de ecuaciones Ax=b usando el metodo de Jacobi con n iteraciones
    '''
    if x is None:
        x = np.zeros_like(b)

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    L_U = L + U

    for i in range(n):
        #if((i % 10) == 0):
        #    print("Iteration:", i, " x: ", x.T)
        
        x0 = x.copy()
        x = np.linalg.inv(D).dot(b - (L_U).dot(x))
        
        if omega != 0:
            r = x - x0
            x = x + omega*r
        
        # Parada
        if(norm(x,x0) < e):
            if prnt:
                print("Convergencia en la iteración:", i+1, "con un error de", e)
            break
            
    error = norm(x,x0)
    
    return x#, error, i


def gauss_seidel(A, b, n, x=None, omega=0, e=0.0001, prnt=True):
    '''
    Resuelve el sistema de ecuaciones Ax=b usando el metodo de Gauss-Seidel con n iteraciones
    '''
    
    if x is None:
        x = np.zeros_like(b)
    
    L = np.tril(A)
    U = A - L
    
    for i in range(n):
        #if((i % 10) == 0):
        #    print("Iteration:", i, " x: ", x.T)

        x0 = x.copy()
        x = np.linalg.inv(L).dot(b - U.dot(x))
        
        if omega != 0:
            r = x - x0
            x = x + omega*r
        
        # Parada
        if(norm(x,x0) < e):
            if prnt:
                print("Convergencia en la iteración:", i+1, "con un error de", e)
            break
    
    error = norm(x,x0)
    
    return x#, error, i


def krylov_gradient_descent(A, b, n, x=None, omega=0, e=0.0001, prnt=True):
    '''
    Resuelve el sistema de ecuaciones Ax=b usando el metodo de Krylov del gradiente descendiente
    '''
    
    if x is None:
        x = np.zeros_like(b)
        
    for i in range(n):
        
        x0 = x.copy()
        
        r = b - A.dot(x0)
        alpha = (r.dot(r))/(r.dot(A.dot(r)))
        x = x0 + alpha*r
        
        if omega != 0:
            r_w = x - x0
            x = x + omega*r_w
            
        #Parada
        if(norm(x,x0) < e):
            if prnt:
                print("Convergencia en la iteración:", i+1, "con un error de ", e)
            break
    
    error = norm(x,x0)
    
    return x#, error, i

def generate_functions_vector(grid, x):
    '''
    Genera el vector de funciones F(x) para una malla dada, y lo evalúa con un vector x dado
    '''
    h = grid.shape[0] - 1 
    w = grid.shape[1] - 1
    
    n_of_equations = count_values(grid)
    f_vector = np.zeros(n_of_equations)
   
    eq_codes = numerate_variables(grid)
    #print(eq_codes)
   
    
    left = 0
    right = 0
    up = 0
    down = 0
    
    for i in range(1, h+1): # i es Y y j es x
        for j in range(1, w+1):
            if not is_border(grid, i, j):

                key = get_key(i,j)
                pos = eq_codes[key]               
                
                #left
                if not is_border(grid, i,j-1):
                    
                    left_code = eq_codes[get_key(i,j-1)]
                    left = x[left_code]

                else:
                    left = grid[i,j-1]
                
                #right
                if not is_border(grid, i,j+1):
                    right_code = eq_codes[get_key(i,j+1)]
                    right = x[right_code]
                else:
                    right = grid[i,j+1]
                    
                #down
                if not is_border(grid, i+1, j):
                    down_code = eq_codes[get_key(i+1,j)]
                    down = x[down_code]
                else:
                    down = grid[i+1,j]
                    
                #up
                if not is_border(grid, i-1, j):
                    up_code = eq_codes[get_key(i-1,j)]
                    up = x[up_code]
                else:
                    up = grid[i-1,j]
                    
                    
                 
                    
                center = x[pos]
                
                value = -4*center + left + right + up + down - 0.5*center*right + 0.5*center*left + 0.5  
                
                f_vector[pos] = value
                
    return f_vector



def generate_jacobian(grid, x):
    '''
    Para una malla dada, genera la matriz jacobiana de funciones, y la evalúa con un vector x dado
    '''
    h = grid.shape[0] - 1
    w = grid.shape[1] - 1
    
    n_of_equations = count_values(grid)
    j_grid = np.zeros((n_of_equations, n_of_equations))
    
    eq_codes = numerate_variables(grid)
    #print(eq_codes)
    
    left = 0
    right = 0
    up = 0
    down = 0
    
    for i in range(1, h+1): # i es Y y j es X 
        for j in range(1, w+1):
            if not is_border(grid, i, j):

                key = get_key(i,j)
                pos = eq_codes[key]               
                
                j_grid[pos][pos] = -4-0.5*x[right]+0.5*x[left]
                
                #left
                if not is_border(grid, i,j-1):
                    left = eq_codes[get_key(i,j-1)]
                    j_grid[pos][left] = 1+0.5*x[pos]
                    
                #right
                if not is_border(grid, i,j+1):
                    right = eq_codes[get_key(i,j+1)]
                    #print(right)
                    j_grid[pos][right] = 1-0.5*x[pos]
                    
                #down
                if not is_border(grid, i+1, j):
                    down = eq_codes[get_key(i+1,j)]
                    j_grid[pos][down] = 1
                    
                #up
                if not is_border(grid, i-1, j):
                    up = eq_codes[get_key(i-1,j)]
                    j_grid[pos][up] = 1
                
                #print(i,j, "pos:", pos)
                #for k in range(h*w):
                #    eq_grid[pos][k]=k+1
                
    return j_grid

def newton_raphson(grid, x, n, method="jacobi", e=0.0001, prnt=True):
    '''
    Resuelve un sistema de ecuaciones no lineal usando el metodo de Newton con n iteraciones
    
    Para el metodo x* = x - J^-1F(x) se toma H = -J^-1F(x) y se resuelve JH=-F
    '''
    
    methods = {
        "jacobi":jacobi,
        "gauss-seidel":gauss_seidel
    }
    
    for i in range(n):
        
        jacobian = generate_jacobian(grid, x)
        f = generate_functions_vector(grid, x)
        
        h = methods[method](jacobian, -f, 200, prnt=False)
        
        x0 = x.copy()
        
        if h.shape != x.shape:
            h = np.zeros_like(x)
        
        x = x + h
        
        # Punto de parada
        if norm(x, x0) < e:
            if prnt:
                print("Convergencia en la iteración:", i+1, "con un error de", e)
            break
            
    return x
# Se considera frontera los bordes de la malla y los bordes de las vigas
def is_border(grid, i, j):
    '''
    Retorna True si la posicion dada en la malla, es una condicion de frontera
    '''
    grid_h = grid.shape[0]
    grid_w = grid.shape[1]
    
    if (i >= (grid_h-1)) or (j >= (grid_w-1)) or (i <= 0) or (j <= 0):
        return True
    elif (grid[i+1,j] == -1) or (grid[i-1,j] == -1) or (grid[i,j-1] == -1) or (grid[i,j+1] == -1):
        return True
    elif (grid[i+1,j+1] == -1) or (grid[i-1,j+1] == -1) or (grid[i+1,j-1] == -1) or (grid[i-1,j-1] == -1):
        return True
    else:
        return False


def count_values(grid):
    '''
    Cuenta la cantidad de valores de una malla que no son condiciones de frontera
    '''
    counter = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not is_border(grid, i, j):
                counter += 1
    return counter


def numerate_variables(grid):
    '''
    Crea un diccionario con códigos correspondientes a las posiciones de cada cuadro y su respectivo número de variable
    '''
    counter = 0
    variables = {}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not is_border(grid, i, j):
                key = str(i) + "_" + str(j)
                variables[key] = counter
                counter +=1
    return variables


def get_key(i,j):
    '''
    Para una posición dada i,j retorna la llave para acceder a su valor numérico en un diccionario
    '''
    return str(i) + "_" + str(j)


def get_i_j_from_key(key):
    '''
    Retorna los valores i,j de una llave
    '''
    values = key.split("_")
    return int(values[0]), int(values[1])

# Define la malla para matriz_vel_x
grid_vel_x = copy.deepcopy(matriz_vel_x)


# Define el vector x inicial para matriz_vel_x (debe ser un vector de ceros)
x_vel_x = np.zeros(grid_vel_x.size)

# Llama a la función newton_raphson para resolver la matriz matriz_vel_x
sol_vel_x = newton_raphson(grid_vel_x, x_vel_x, n=50)

# Imprime la solución
print("Solución para matriz_vel_x:")
print(sol_vel_x)