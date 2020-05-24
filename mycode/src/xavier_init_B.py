# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:36:12 2020

@author: nastavirs
"""
import numpy as np
import tensorflow as tf
def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)