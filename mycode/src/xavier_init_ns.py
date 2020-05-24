# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:31:14 2020

@author: nastavirs
"""
import tensorflow as tf
import numpy as np
def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)