import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import brentq
from hardpoint_io import dict_FL, dict_FR,d1,d2,d3,k_U,k_L

#the rodrigues function

def rodrigues(theta_U,k_U,UBJ_stat,arm):
    UBJ_rel=(UBJ_stat-dict_FL[arm])
    Kx=np.array([[0,-k_U[2],k_U[1]],[k_U[2],0,-k_U[0]],[-k_U[1],k_U[0],0]])
    Kx2=((Kx)@(Kx))
    I=np.eye(3)
    R=(I+(Kx*(np.sin(theta_U)))+(Kx2*(1-np.cos(theta_U))))
    return (R@UBJ_rel+dict_FL[arm])

def theta_L_solver(UBJ_curr,LBJ_stat,k_L,seed,bracket,arm):

    def residual(theta_L):
        return (np.linalg.norm(UBJ_curr-rodrigues(theta_L,k_L,LBJ_stat,arm))-d1)
    
    try:
        root=brentq(residual,seed-bracket,seed+bracket,xtol=1e-10)
    except:
        try:
            root=brentq(residual,seed-(bracket*2),seed+(bracket*2),xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed} limits")
    
    LBJ_curr=rodrigues(root,k_L,LBJ_stat,arm)
    return LBJ_curr