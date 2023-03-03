#author Gerbaud FLorent
#Algorithme LU
#03/03/2023

import numpy as np
import math
import matplotlib.pyplot
from copy import deepcopy

#Méthode qui effectue la décomposition de la matrice LU sans pivot
#input : A la matrice a decomposer
#L la matrice  infèrieur dans la décomposition LU qui contient les alpha pour éliminer les pivot succesif a chaque étape
#U la matrice triangulaire supérieur après avoir effectué les élimination par pivot succesif
def factLu(A):
    
    dim=A.shape[1]
    L = np.identity(dim)
    U=deepcopy(A)
    
    for k in range(0, dim):
        for j in range(k+1, dim):
            alpha=U[j,k]/U[k,k]
            U[j,:]=U[j,:]-alpha*U[k,:] #application de la formule du pivot
            L[j,k]=alpha 
    return (L,U)
   
#Méthode qui effectue l'algorithme de descente
#input : L la matrice triangulaire inférieur, et b un vecteur colonne 
def descente(L,b):
    dim=L.shape[1]
    L1=np.copy(L)
    Y=np.zeros((dim,1))
    Y[0]=b[0]/L1[1,1] #initialisation algo
    somme=0
    for i in range(1,dim):
        for k in range(0,dim-1):
            somme=somme+L1[i,k]*Y[k] #effectue la somme des aik*yk
        Y[i]=(b[i]-somme)/L1[i,i] #puis on calcule le yk+1
        somme=0
    return Y
   
#Méthode qui effectue l'algorithme de remontée
#input : L la matrice triangulaire supérieur, et y un vecteur colonne  
#exactement le meme procédé que descente sauf que cette fois ci on part du bas pour remonter jusque en haut
def remonte(U,y):
    dim=U.shape[1]
    U1=np.copy(U)
    y1=np.copy(y)
    X=np.zeros((dim,1))
    X[dim-1]=y1[dim-1]/U1[dim-1,dim-1] #initialisation algorithme
    somme=0
    for i in range(dim-1,-1,-1):
        for k in range(i+1,dim):
            somme=somme+U1[i,k]*X[k]
        X[i]=(y1[i]-somme)/U1[i,i]
        somme=0
    return X
    
def factLuPivot(A):
    
    #dim : dimension de la matrice (int)
    #P : matrice de permutation globale
    #L : matrice de la decomposition LU 
    #U : matrice de la decomposition LU
    
    dim=A.shape[1]
    P=np.eye(dim)
    L = np.eye(dim)
    U=deepcopy(A)
    #construction de U, M(n), P 
    for k in range(0,dim-1):
        #formule calcul de la ligne que l'n va intervertir pour le pivot
        k0=np.argmax(abs(U[k:dim,k]))
        k0=k0+k #demander au prof pq ?
        #intervertion des lignes
        Pk=np.eye(dim,dim)
        if(k0!=k):
            
            u=deepcopy(U[k0,k:dim])
            U[k0,k:dim]=U[k,k:dim]
            U[k,k:dim]=u
            #construction de P 
            Pk[k0,k]=1
            Pk[k,k0]=1
            Pk[k,k]=0
            Pk[k0,k0]=0
            P=np.dot(Pk,P) #aj matrice permutation globale
        
        Mp=np.eye(dim)
        for j in range(k+1,dim):
            alpha=U[j,k]/U[k,k]
            U[j,:]=U[j,:]-alpha*U[k,:]
            #matrice triangulaire inf qui contient les valeurs des alpha pour supprimer les coefficientsen dessous du pivot
            Mp[j,k]=alpha
            #construction de mon M(n)
        L=np.dot(L,np.dot(np.transpose(Pk),Mp))
        
    L=np.dot(P,L)
    return (L,U,P)

############################# test ####################################

#Matrice A et vecteur b afin de résoudre Ax=b
A = np.array([[-2.,1.,-1.,1.],[2.,0.,4.,-3.],[-4.,-1.,-12.,9.],[-2.,1.,1.,-4.]])
b=np.array([[1.5],[4],[-14],[-6.5]])

############################Test avec la méthode LU sans changement de pivot ###########################################

print("Test avec la méthode LU sans changement de pivot")
(L,U)=factLu(A)
y=descente(L,b)
x=remonte(U,y)
print('L=',L)
print('U=',U)
print("la solution est x=", x)
print("")

############################Test de la méthode LU avec changement de pivot ###########################################

print("Test avec la méthode LU avec changement de pivot")
(L,U,P)=factLuPivot(A)
print('L=',L)
print('U=',U)
print('P=',P)

print("P.A=",np.dot(P,A))
print("L.U=",np.dot(L,U))
bP=np.dot(P,b)
yP=descente(L,bP)
xP=remonte(U,yP)
print("la solution est x = ", xP)
