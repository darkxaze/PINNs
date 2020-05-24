# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:31:37 2020

@author: nastavirs
"""

import tensorflow as tf
import numpy as np
def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %.3e, l1: %.3f, l2: %.5f' % (loss, lambda_1, lambda_2))