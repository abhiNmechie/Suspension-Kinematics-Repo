import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import brentq
from hardpoint_io import dict_FL, dict_FR,d1,d2,d3,k_U_Left,k_L_Left

def rodrigues_front(theta,k,BJ_stat_rel,vector_rel_origin):
    Kx=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    Kx2=((Kx)@(Kx))
    I=np.eye(3)
    R=(I+(Kx*(np.sin(theta)))+(Kx2*(1-np.cos(theta))))
    return (R@BJ_stat_rel+vector_rel_origin)                             

def upper_front(theta_U,k_U,BJ_stat_rel,vector_rel_origin):
    UBJ_curr=(rodrigues_front(theta_U,k_U,BJ_stat_rel)+vector_rel_origin)
    return UBJ_curr                               

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
    return LBJ_curr



def tierod(UBJ_curr,LBJ_curr,UBJ_stat,LBJ_stat,TRO_stat,seed,bracket):
    
    kp_static=(UBJ_stat-LBJ_stat)/(np.linalg.norm(UBJ_stat-LBJ_stat))
    v_static=(TRO_stat-LBJ_stat)
    component=(np.dot(v_static,kp_static))
    radius=np.linalg.norm(v_static-(component*kp_static))
    
    #seed initialization
    #tmp=np.array([1,0,0])
    #x=(tmp-((np.dot(tmp,kp_static))/np.linalg.norm(kp_static)))  
    #e1=(x/np.linalg.norm(x))
    #a=(v_static-component*kp_static)
    #y=((np.dot(a,e1))/((np.linalg.norm(a))*(np.linalg.norm(e1))))
    #z=np.arccos(y)
    #print(z)
    #quit()


    kp_curr=(UBJ_curr-LBJ_curr)/(np.linalg.norm(UBJ_curr-LBJ_curr))
    centre_curr=((component*kp_curr)+LBJ_curr)


    tmp=np.array([1,0,0])
    if np.dot(kp_curr,tmp)>0.9 or np.dot(kp_curr,tmp)<-0.9:
        tmp=np.array([0,1,0])
    
    x=(tmp-((np.dot(tmp,kp_curr))*kp_curr)/(np.linalg.norm(kp_curr)))  
    e1=(x/np.linalg.norm(x))
    e2=np.cross(kp_curr,e1)

    def tro_post(phi,e1,e2,radius,centre):
        return (centre+((radius*np.cos(phi))*e1)+((radius*np.sin(phi))*e2))
    
    def residual(phi):
        return (np.linalg.norm(tro_post(phi,e1,e2,radius,centre_curr)-dict_FL['TRI'])-d2)
    try:
          root=brentq(residual,seed-bracket,seed+bracket,xtol=1e-10)
    except:
        try:
              root=brentq(residual,seed-bracket*2,seed+bracket*2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed} limits")
        
    TRO_curr=tro_post(root,e1,e2,radius,centre_curr)
    return TRO_curr


print(tierod(dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],0.400918197828729,0.1))


def triad_transform(UBJ_curr,LBJ_curr,TRO_curr,UBJ_stat,LBJ_stat,TRO_stat,vector_rel_lbj):

    e1_stat=(UBJ_stat-LBJ_stat)/(np.linalg.norm(UBJ_stat-LBJ_stat))
    v=(TRO_stat-LBJ_stat)
    e2_stat=(v-np.dot(v,e1_stat)*e1_stat)/(np.linalg.norm(v-np.dot(v,e1_stat)*e1_stat))
    e3_stat=np.cross(e2_stat,e1_stat)
    M1=np.array([[e1_stat],[e2_stat],[e3_stat]])

    e1_curr=(UBJ_curr-LBJ_curr)/(np.linalg.norm(UBJ_curr-LBJ_curr))
    v_curr=(TRO_curr-LBJ_curr)
    e2_curr=(v_curr-np.dot(v_curr,e1_curr)*e1_curr)/(np.linalg.norm(v_curr-np.dot(v_curr,e1_curr)*e1_curr))
    e3_curr=np.cross(e2_curr,e1_curr)
    M2=np.array([[e1_curr],[e2_curr],[e3_curr]])

    return (((M2.T)@(M1)@vector_rel_lbj)+LBJ_stat)