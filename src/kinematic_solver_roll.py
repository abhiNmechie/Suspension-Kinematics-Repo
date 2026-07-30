import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.optimize import fsolve
from hardpoint_io import dict_FL, dict_FR, dict_RL, dict_RR, FL_d1, FL_d2, FL_d3, FR_d1, FR_d2, FR_d3, RL_d1, RL_d2, RR_d3, RR_d1, RR_d2, RR_d3, FL_k_L,FL_k_U,FR_k_L,FR_k_U,RL_k_L,RL_k_U,RR_k_L,RR_k_U

def rodrigues(theta,BJ_stat_rel,k,vector_rel_origin):
    I=np.identity(3)
    kx=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    kx2=kx@kx
    R=I+kx*np.sin(theta)+(kx2)*(1-np.cos(theta))
    return ((R@BJ_stat_rel)+vector_rel_origin)

def upper(theta_U,UBJ_stat_rel,vector_rel_origin,k_U):
    UBJ_curr=rodrigues(theta_U,UBJ_stat_rel,k_U,vector_rel_origin)
    return UBJ_curr

def lower(UBJ_curr,LBJ_stat_rel,vector_rel_origin,k_L,seed,const_dist):
    def residual(theta_L):
        return (np.linalg.norm(UBJ_curr-rodrigues(theta_L,LBJ_stat_rel,k_L,vector_rel_origin))-const_dist)
    root=fsolve(residual,seed)[0]
    LBJ_curr=rodrigues(root,LBJ_stat_rel,k_L,vector_rel_origin)
    return (LBJ_curr,root)

def tierod()