import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import brentq
from hardpoint_io import dict_FL, dict_FR,d1,d2,d3,k_U_Left,k_L_Left

#the rodrigues function

#the front setup
def rodrigues_front(theta,k,BJ_stat_rel,vector_rel_origin):
    Kx=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    Kx2=((Kx)@(Kx))
    I=np.eye(3)
    R=(I+(Kx*(np.sin(theta)))+(Kx2*(1-np.cos(theta))))
    return (R@BJ_stat_rel+vector_rel_origin)                              #IT RETURNS ONLY RELATIVE VALUE NOTEEE

def upper_front(theta_U,k_U,BJ_stat_rel,vector_rel_origin):
    UBJ_curr=(rodrigues_front(theta_U,k_U,BJ_stat_rel)+vector_rel_origin)
    return UBJ_curr                                 ##IT ALSO RETURNS RELATIVE VALUE NOTEE

def lower_front(UBJ_curr,LBJ_stat_rel,vector_rel_origin,k_lower_side,seed,bracket):
    def residual(theta_L):
        return (np.linalg.norm(UBJ_curr-rodrigues_front(theta_L,k_lower_side,LBJ_stat_rel,vector_rel_origin))-d1)
    
    try:
        root=brentq(residual,seed-bracket,seed+bracket,xtol=1e-10)
    except:
        try:
            root=brentq(residual,seed-(bracket*2),seed+(bracket*2),xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed} limits")
    
    LBJ_curr=rodrigues_front(root,k_L_Left,LBJ_stat_rel,vector_rel_origin)
    print(root)
    return LBJ_curr


print(lower_front(dict_FL['UBJ'],(dict_FL['LBJ']-dict_FL['LF']),dict_FL['LF'],k_L_Left,0.0,0.05))

def tierod(UBJ_curr,LBJ_curr,UBJ_stat,LBJ_stat,TRO_stat,seed,bracket):
    kp_static=(UBJ_stat-LBJ_stat)/(np.linalg.norm(UBJ_stat-LBJ_stat))
    v_static=(TRO_stat-LBJ_stat)
    component=(np.dot(v_static,kp_static))