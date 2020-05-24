# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:37:18 2020

@author: nastavirs
"""
import numpy as np
def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, np.exp(lambda_2)))