# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:54:01 2019

@author: Marsupilami
"""
from pandas import DataFrame
from numpy import arange
from random import seed,shuffle

# pyDataFrame object
class PyDataFrame(DataFrame):
    
    features = []
    target = []
    rate = 0
    seed = None
    idx_train = []
    idx_test = []
    	
    def __init__(self,features,target,rate=0.3,seed=None,**kwargs):
        super(PyDataFrame,self).__init__(**kwargs)
        self.features = features
        self.target = target
        self.rate = rate
        self.seed = seed
        self.shuffle_index(self.seed)
                
    
    def X_train(self,feature = None):
        return self[self.features if feature == None else feature].loc[self.idx_train]
    
    def y_train(self,target = None):
        return self[self.target if target == None else target].loc[self.idx_train]
    
    def X_test(self,feature = None):
        return self[self.features if feature == None else feature].loc[self.idx_test]
    
    def y_test(self,target = None):
        return self[self.target if target == None else target].loc[self.idx_test]
    
    def shuffle_index(self,_seed=None):
        """
        Return two Numpy 1D array from one combinaison of shuffle indexes taking values between [0;N].
        First array containt [:rate] and the second [rate:].
        This function is design for separate a dataset in train and test set with a rate.
        
        N : maximum of index (int)
        rate : separation 
        
        return : a typle with int nympy arrays
        21FEB19.
        """
        
        # select random state
        seed(_seed if _seed != None else self.seed)
        rate = self.rate
        
        output = False
        if 0 <= rate <= 1:
            index = arange(0,self.shape[0],1)
            shuffle(index)
            output = (index[:int(len(index)*rate)],index[int(len(index)*rate):])
        else : 
            raise Exception("Rate is not containt between [0;1]")
        
        self.idx_train = output[0]
        self.idx_test = output[1]
        
        return None