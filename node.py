# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:47:10 2019

@author: X188212
"""

from numpy import arange
from sklearn.linear_model import LinearRegression

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
    eps = None
    
    def __init__(self,T,eps,build = True):
        self.eps = eps
        self.T = T
        if build:
            self.root = self.generateTree(T,eps)[-1][0]
        else:
            self.root = self.findRoot(self.T)

    def makeAFather(self,left,right,value):
        t = left.t + 1
        h = left.h + self.eps
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
        

    def convertBinMsg(self,binMsg,verbose = False):
        self.root.value = binMsg[0]
        self._convertBinMsg(binMsg.copy(),self.root,verbose)

    def _convertBinMsg(self,binMsg,root,verbose = False):
        currentNode = root
        self.coord = [currentNode.extract()]
        for x in binMsg:
            if verbose : print(currentNode)
            if x == 0:
                currentNode = self._findLeftChild(currentNode)
                self.coord.append(currentNode.extract())
            elif x == 1:
                currentNode = self._findRightChild(currentNode)
                self.coord.append(currentNode.extract())
            else:
                raise ValueError("Message element is not binary: %s"%x)
        return 1
        
    def extractCoords(self,pathType = 'preOrder'):
        self.path_coords = []
        return self._extractCoord(self.root,pathType)
    
    def _extractCoord(self,node,pathType):
            
        if pathType == "inOrder":

            if node.left != None: self._extractCoord(node.left,pathType)
            self.raw_coords.append(node.extract())
            if node.right != None: self._extractCoord(node.right,pathType)

        elif pathType == "preOrder":

            self.raw_coords.append(node.extract())
            if node.left != None: self._extractCoord(node.left,pathType)
            if node.right != None: self._extractCoord(node.right,pathType)

        elif pathType == "postOrder":

            if node.left != None: self._extractCoord(node.left,pathType)
            if node.right != None: self._extractCoord(node.right,pathType)
            self.raw_coords.append(node.extract())

        else:
            pass
        return 1
    
    def findRoot(self,T):
        return Node(T,2**T-1+T)
    
    def _findChildren(self,node):
        t,h = node.extract()
        out = None
        if t-1 >= 0:
            out = (Node(t-1,h-1,0),Node(t-1,h-1-2**(t-1),1))
        return out
    
    def _findLeftChild(self,node):
        out = None
        t,h = node.extract()
        if t - 1 >= 0:
            out = Node(t-1,h-1,0)
        return out
    
    def _findRightChild(self,node):
        out = None
        t,h = node.extract()
        if t - 1 >= 0:
            out = Node(t-1,h-1-2**(t-1),1)
        return out