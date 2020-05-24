# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:37:36 2020

@author: nastavirs
"""
import numpy as np
import tensorflow as tf
def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star, f_star