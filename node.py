# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:47:10 2019

@author: X188212
"""

from numpy import arange

class Node: 
    
    def __init__(self,t,h,value=None,left=None,right=None,father=None):
        self.t = t
        self.h = h
        self.affect(father,left,right)
        self.value = value
        
    def affect(self,father,left,right):
        self.left = left
        self.right = right
        self.father = father
    
    def extract(self):
        return (self.t,self.h)
    
    def __repr__(self):
        return "<Node(%s / %s / %s)>"%(self.t,self.h,self.value)

class Tree:
    
    limit = 4
    
    def __init__(self,T,eps):
        self.root = self.generateTree(T,eps)[-1][0]

    def makeAFather(self,left,right,value):
        t = left.t + 1
        h = left.h + eps
        father = Node(t,h,value,left,right,None)
        left.father = father
        right.father = father
        return father
    
    def makeFatherLevel(self,current_level):
        father_level = []
        k = 0
        while len(current_level) > 0:
            left = current_level.pop(0)
            right = current_level.pop(0)
            father = self.makeAFather(left,right,k)
            father_level.append(father)
            k = 0 if k == 1 else 1
        return father_level
    
    def generateT_level(self,T,eps):
        H = 2**T
        k = 0
        level_zero = []
        for h in arange(0,H*eps,eps):
            level_zero.append(Node(0,h,k))
            k = 0 if k == 1 else 1
        return level_zero
    
    def generateTree(self,T,eps):
        leves = self.generateT_level(T,eps)[::-1]
        tree = []
        current_level = leves
        while len(current_level) >= 2:
            tree.append(current_level.copy())
            current_level = self.makeFatherLevel(current_level)
        tree.append(current_level.copy())
        return tree
    
    def preOrder(self):
        self._preOrder(self.root,0)
    
    def _preOrder(self,node,level):
        print(node)
        if level <= self.limit:
            if node.left != None:
                self._preOrder(node.left,level+1)
            if node.right != None:
                self._preOrder(node.right,level+1)
        else:
            pass
        
    def convertBinMsg(self,binMsg):
        currentNode = self.root
        self.coord = [currentNode.extract()]
        for x in binMsg:
            if x == 0:
                currentNode = currentNode.left
            else:
                currentNode = currentNode.right
            self.coord.append(currentNode.extract())
        return self.coord

    def convertCurveToBinmsg(self,coefs,intercept,T):
        lr2 = LinearRegression()
        lr2.coef_ = coefs
        lr2.intercept_ = intercept
        
        t = T
        
        
        
#%%  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy

msg = np.random.randint(0,2,17)

T = len(msg)
eps = 1
tree = Tree(T,eps)

X,Y = zip(*tree.convertBinMsg(msg))

X = np.array(X)
Y = np.array(Y)





#%%
plt.plot(X,Y,'.',label = "data")

# f1 fit
f1 = lambda x,a,b : a*np.exp(b*x)
results = scipy.optimize.curve_fit(f1,  X,  Y)
a,b = results[0]
e1 = np.sqrt(np.sum((Y - f1(X,a,b))**2))
plt.plot(X,f1(X,a,b),label="f1 : %s"%e1)


# f2 fit
f2 = lambda x,a,b,c,d : c*np.exp(a*x+b)+d
results = scipy.optimize.curve_fit(f2,  X,  Y)
a,b,c,d = results[0]
e2 = np.sqrt(np.sum((Y - f2(X,a,b,c,d))**2))
plt.plot(X,f2(X,a,b,c,d),label="f2 : %s"%e2)


# f2 fit with more parameters
f22 = lambda x,a,b,c,d : c*np.exp(a*x+b)+d
results = scipy.optimize.curve_fit(f22,  X,  Y,p0 =(0.8,0.8,0.8,0.8))
a,b,c,d = results[0]
e2 = np.sqrt(np.sum((Y - f22(X,a,b,c,d))**2))
plt.plot(X,f22(X,a,b,c,d),label="f22 : %s"%e2)


# f3
f3 = lambda x,a,c,d : c*np.exp(a*x)+d
results = scipy.optimize.curve_fit(f3,  X,  Y)
a,c,d = results[0]
e3 = np.sqrt(np.sum((Y - f3(X,a,c,d))**2))
plt.plot(X,f3(X,a,c,d),label="f3 : %s"%e3)


plt.legend()

#%%

# f1 fit
f1 = lambda x,a,b : a*np.exp(b*x)
results = scipy.optimize.curve_fit(f1,  X,  Y)
a,b = results[0]
plt.plot(X,(Y-f1(X,a,b))**2,label="f1 errors")


# f2 fit
f2 = lambda x,a,b,c,d : c*np.exp(a*x+b)+d
results = scipy.optimize.curve_fit(f2,  X,  Y)
a,b,c,d = results[0]
plt.plot(X,(Y-f2(X,a,b,c,d))**2,label="f2 errors")


# f2 fit with more parameters
f22 = lambda x,a,b,c,d : c*np.exp(a*x+b)+d
results = scipy.optimize.curve_fit(f22,  X,  Y,p0 =(0.8,0.8,0.8,0.8))
a,b,c,d = results[0]
plt.plot(X,(Y-f22(X,a,b,c,d))**2,label="f22 errors")


# f3
f3 = lambda x,a,c,d : c*np.exp(a*x)+d
results = scipy.optimize.curve_fit(f3,  X,  Y)
a,c,d = results[0]
e3 = np.sqrt(np.sum((Y - f3(X,a,c,d))**2))
plt.plot(X,(Y-f3(X,a,c,d))**2,label="f3 errors")


plt.legend()




#%%
deg = 5
p = PolynomialFeatures(degree = deg).fit(X.reshape(-1,1),Y.reshape(-1,1))
X_transform = p.transform(X.reshape(-1,1))
lr = LinearRegression().fit(X_transform,Y.reshape(-1,1))

Y_pred = lr.predict(X_transform)


plt.plot(X,Y,'.')
plt.plot(X,Y_pred)

for i in range(5):
    print("%s"%i,end = "\r")
    msg = np.random.randint(0,2,10)
    
    T = len(msg)
    eps = 10E-4
    tree = Tree(T,eps)
    
    X,Y = zip(*tree.convertBinMsg(msg))
    
    X = np.array(X)
    Y = np.array(Y)
    
    deg = 5
    p = PolynomialFeatures(degree = deg).fit(X.reshape(-1,1),Y.reshape(-1,1))
    X_transform = p.transform(X.reshape(-1,1))
    lr = LinearRegression().fit(X_transform,Y.reshape(-1,1))
    
    Y_pred = lr.predict(X_transform)
    
    
    plt.plot(X,Y,'.',label="%s"%i)
    plt.plot(X,Y_pred,label="%s line"%i)
plt.legend()


a = 1/2
b = 1
c = 1
f = lambda x : np.exp(a*x+b)+c

x = np.arange(10)
y = f(x)

plt.plot(x,y)





