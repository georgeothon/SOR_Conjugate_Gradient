#Bibliotecas
import numpy as np
from scipy.io import mmread #Le a matriz
import time

def norma_infinita(x, x_2):
    
    #x_k
    norm_x = max(abs(x))
    
    return max(abs(x_2 - x))/norm_x

def SOR(A, b, n, w=1.94, ite_max = 10**5, tol = 10**(-9)):
    
    #Formato COO
    row = A.row
    col = A.col
    data = A.data
    
    n = len(b)
    x = np.zeros(n)
    XO=np.zeros(n)
    
    k = 1
    
    while k <= ite_max :
        
        for i in range(n):
            
            #elementos na linha i
            indicador_i = row == i
            
            #elemento a_ii
            a_ii = data[np.logical_and(indicador_i,col == i)][0]
            
            
            #separa elementos que vamos multiplicar pelo vetor x e pelo vetor XO
            j_menor_i = np.logical_and( indicador_i, col < i )
            j_maior_i = np.logical_and( indicador_i, col > i )
            
            data_x = data[j_menor_i]
            data_XO = data[j_maior_i] 
            
            #Escolhe os elementos de x que serão multiplicador
            indicador_x = col[j_menor_i]
            indicador_XO = col[j_maior_i]
            
            vetor_x = np.array([ x[m] for m in indicador_x ])
            vetor_XO = np.array([ XO[m] for m in indicador_XO ])
            
            #calcula x_i
            x[i] = (1-w)*XO[i] + (1/a_ii)*(w*( - np.sum( data_x * vetor_x ) - np.sum( data_XO * vetor_XO ) + b[i]))

            
        #verifica se atingiu o critério
        if norma_infinita(x,XO) < tol:
            return x, k
        

        #próxima iteração
        k += 1
        
        XO = x.copy()
        
    return x, k-1

def mul_esparsa(A, v):
    
    #Multiplica a matriz esparsa A pelo vetor v
    
    row = A.row
    col = A.col
    data = A.data
    
    n = len(v)
    
    u = np.zeros(n)
    
    for i in range(n):
        
        indicador = row == i
        v_i = v[col[indicador]]
        A_i = data[indicador]
        u[i] = np.sum(A_i * v_i)
        
    return u    


def Conjugate_Gradient(A, b, x, tol=10**-9):
    
    n = len(b)
    
    r = b - mul_esparsa(A,x)
    v = r.copy()
    alfa = np.dot(v,v)
    
    k = 1
    
    while k <= n:
        
        #Calcula a direção e o modulo para incrementar no x e no residuo
        u = mul_esparsa(A,v)
        t = alfa/np.sum(v*u)
        
        
        x = x + t*v
        r = r - t*u
                
        beta = np.sum(r*r)
        
        #Verica se atingiu a tolerância
        if beta < tol:
            return x, r, k
        
        #muda o vetor direção
        s = beta/alfa
        v = r + s*v
        alfa = beta
        
        k += 1
        
    return x, r, k

def main():
    
    #Le a matrix no formato COO

    path = ['C:/Users/georg/OneDrive/Área de Trabalho/10349978_T2/bcsstk01.mtx' ,
    'C:/Users/georg/OneDrive/Área de Trabalho/10349978_T2/nos4.mtx',
    'C:/Users/georg/OneDrive/Área de Trabalho/10349978_T2/mesh3em5.mtx',
    'C:/Users/georg/OneDrive/Área de Trabalho/10349978_T2/nos5.mtx']
    
    w_otimo = [1.9, 1.9, 1.1, 1.9]
    
    n_nulos = [17.4, 5.9, 0.022, 2.4]
    
    
    for i in range(len(path)):
        
        A = mmread(path[i])

        #como a matriz é simetrica, e positiva definida temos que n é o maior indice
        n = max(A.row) + 1
    
        b = mul_esparsa(A, np.ones(n))
        
        
        print('\n\n==============================================\n')
        print(f'Matriz {n}x{n} com {n_nulos[i]}% não nulos \n ')
        print('==============================================\n\n')
        
        print('=============== Algoritmo SOR ==================\n')
        
        ti_SOR = time.time()
        x_SOR, k_SOR = SOR(A, b, n, w=w_otimo[i], tol=10**-9)
        tf_SOR = time.time()
        
        print(f'Resultado:\n {x_SOR} \n iterações: {k_SOR} \n Tempo em segundos: {tf_SOR-ti_SOR} \n')
        
        
        print('=========== Algoritmo Conjugate Gradient ============ \n')

        ti_CG = time.time()
        x_CG, r, k_CG = Conjugate_Gradient(A, b, np.zeros(n))
        tf_CG = time.time()
        
        print(f'Resultado:\n {x_CG}\n Resíduo:\n {r} \n iterações: {k_CG} \n Tempo em segundos: {tf_CG-ti_CG} \n\n')
        
    
main()





