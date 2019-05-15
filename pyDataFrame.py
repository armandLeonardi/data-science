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
    """
    Class heritate of DataFrame from pandas module.
    
    It perform an automatic separation between train set and test set.
    When your init a PyDataFrame you plug a set of data, select which features will be in
    thes inputs of yours problem and the outputs. A schuffle function randomly select
    indexes save the result.
    
    
    Here an example :
        
    >>> pdf = PyDataFrame(data = data_object,
                         features = ["feat1","feat2",...],
                         tagret = ["target1","target2",...],
                         rate = 0.3,
                         seed = 1234)
    
    If you want an other selection for train and test set, use 'shuffle_index' function again.
    
    
    @attributes 
    
    attribute 1 : features : list of columns names containing features of your problem
    type atribute 1 : list of string

    attribute 2 : target : list of columns which your target, or the features you want to predict.
                            Their can be more than once.
    type attribute 2 : list of string
    
    attribute 3 : rate : sepration ratio
    type attribute 3 : float
    
    attribute 4 : seed : the id of a seed, if you want a replicate separation between train and test
                        set.
    type attribute 4 : int
    
    attribute 5 : list of indexes wich selected for the train set
    type attribute 5 : list of int
    
    attribute 6 : list of indexes wich selected for the test set
    type attribute 6 : list of int
    

    V1 - 21FEB19 - Armand Léonardi
    """
    
    
    features = []
    target = []
    rate = 0
    seed = None
    idx_train = []
    idx_test = []
    	
    def __init__(self,features,target,rate=0.3,seed=None,*args,**kwargs):
        super(PyDataFrame,self).__init__(*args,**kwargs)
        self.features = features
        self.target = target
        self.rate = rate
        self.seed = seed
        self.shuffle_index(self.seed)
                
    
    def X_train(self,feature = None):
        """Return the inputs (features) of the train set"""
        return self[self.features if feature == None else feature].loc[self.idx_train]
    
    def y_train(self,target = None):
        """Return the outputs of the train set"""
        return self[self.target if target == None else target].loc[self.idx_train]
    
    def X_test(self,feature = None):
        """Return the inputs (features) of the test set"""
        return self[self.features if feature == None else feature].loc[self.idx_test]
    
    def y_test(self,target = None):
        """Return the outputs of the test set"""
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
        
        return 1