import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import brentq
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from hardpoint_io import dict_FL,dict_FR,dict_RL, dict_RR,FL_d1,FL_d2,FL_d3,FR_d1,FR_d2,FR_d3,RL_d1,RL_d2,RL_d3,RR_d1,RR_d2,RR_d3,FL_k_L,FL_k_U,FR_k_L,FR_k_U,RL_k_L,RL_k_U,RR_k_L,RR_k_U
np.set_printoptions(suppress=True, precision=8)

#output:
#FL:
out_dict_FL=dict()
for k,v in dict_FL.items():
    out_dict_FL[k]=np.zeros((13,3))
    out_dict_FL[k][6]=v

#RL:
out_dict_RL=dict()
for k,v in dict_RL.items():
    out_dict_RL[k]=np.zeros((13,3))
    out_dict_RL[k][6]=v

#FR:
out_dict_FR=dict()
for k,v in dict_FR.items():
    out_dict_FR[k]=np.zeros((13,3))
    out_dict_FR[k][6]=v

#RR:
out_dict_RR=dict()
for k,v in dict_RR.items():
    out_dict_RR[k]=np.zeros((13,3))
    out_dict_RR[k][6]=v

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

def tierod(UBJ_curr,LBJ_curr,UBJ_stat,LBJ_stat,TRO_stat,seed,TRI_stat,const_dict):
    kp_stat=((UBJ_stat-LBJ_stat)/(np.linlag.norm(UBJ_stat-LBJ_stat)))
    v_stat=(TRO_stat-LBJ_stat)
    centre_stat=(np.dot(v_stat,kp_stat)*kp_stat+LBJ_stat)
    radius=np.linalg.norm(v_stat-np.dot(v_stat,kp_stat)*kp_stat)

    v_curr=(TRO_curr-LBJ_curr)
    kp_curr=((UBJ_curr-LBJ_curr)/(np.linlag.norm(UBJ_curr-LBJ_curr)))
    centre_curr=(np.dot(v_curr,kp_curr)*kp_curr+LBJ_curr)

    vect_tmp=np.array([1,0,0])
    if np.dot(vect_tmp,kp_curr)>0.9 or np.dot(vect_tmp,kp_curr)<(-0.9):
        vect_tmp=np.array([0,1,0])

    e1_curr=((vect_tmp-np.dot(vect_tmp,kp_curr)*kp_curr)/np.linalg.norm(vect_tmp-np.dot(vect_tmp,kp_curr)*kp_curr))
    e2_curr=np.cross(kp_curr,e1_curr)
    def residual(phi):
        return (np.linalg.norm((centre_curr+radius*np.cos(phi)*e1_curr+radius*np.sin(phi)*e2_curr)-TRI_stat)-const_dict)

    root=fsolve(residual,seed)[0]
    TRO_curr=(centre_curr+radius*np.cos(root)*e1_curr+radius*np.sin(root)*e2_curr)

    return (TRO_curr,root)

def seeder(UBJ_stat,LBJ_stat,TRO_stat):
    kp_stat=((UBJ_stat-LBJ_stat)/(np.linalg.norm(UBJ_stat-LBJ_stat)))
    v_stat=(TRO_stat-LBJ_stat)

    vect_tmp=np.array([1,0,0])
    if np.dot(vect_tmp,kp_stat)>0.9 or np.dot(vect_tmp,kp_stat)<(-0.9):
        vect_tmp=np.array([0,1,0])

    e1_stat=((vect_tmp-np.dot(vect_tmp,kp_stat)*kp_stat)/np.linalg.norm(vect_tmp-np.dot(vect_tmp,kp_stat)*kp_stat))
    e2_stat=np.cross(kp_stat,e1_stat)
    seed_int=np.arctan2((np.dot(v_stat,e2_stat)),(np.dot(v_stat,e1_stat)))
    return seed_int

def triad_transform(UBJ_curr,LBJ_curr,TRO_curr,UBJ_stat,LBJ_stat,TRO_stat,vector_stat):
    e1_stat=(UBJ_stat-LBJ_stat)/np.linalg.norm(UBJ_stat-LBJ_stat)
    e2_stat=(TRO_stat-LBJ_stat)/np.linalg.norm(TRO_stat-LBJ_stat)
    e3_stat=np.cross(e1_stat,e2_stat)

    e1_curr=(UBJ_curr-LBJ_curr)/np.linalg.norm(UBJ_curr-LBJ_curr)
    e2_curr=(TRO_curr-LBJ_curr)/np.linalg.norm(TRO_curr-LBJ_curr)
    e3_curr=np.cross(e1_curr,e2_curr)

    M=np.array([e1_stat,e2_stat,e3_stat])
    vector_rel_stat=(vector_stat-LBJ_stat)
    vector_rel_curr=(M.T)@(M)@(vector_rel_stat)

    return (vector_rel_curr+LBJ_curr)

#RC_Height_static:

def cross_point(Fore,Aft,BJ):
    v1=(BJ-Fore)
    v2=(Aft-Fore)/np.linalg.norm(Aft-Fore)
    v3=np.dot(v1,v2)*v2
    return (v3+Fore)

def solver_wishbones(Fore_1,Aft_1,BJ_1,Fore_2,Aft_2,BJ_2):
    cross_pt_1=cross_point(Fore_1,Aft_1,BJ_1)
    A=(cross_pt_1[2]-BJ_1[2])
    B=(BJ_1[1]-cross_pt_1[1])
    C=(-cross_pt_1[1]*BJ_1[2]+cross_pt_1[2]*BJ_1[1])

    cross_pt_2=cross_point(Fore_2,Aft_2,BJ_2)
    D=(cross_pt_2[2]-BJ_2[2])
    E=(BJ_2[1]-cross_pt_2[1])
    F=(-cross_pt_2[1]*BJ_2[2]+cross_pt_2[2]*BJ_2[1])

    y=np.linalg.det(np.array([[C,B],[F,E]]))/np.linalg.det(np.array([[A,B],[D,E]]))
    z=np.linalg.det(np.array([[A,C],[D,F]]))/np.linalg.det(np.array([[A,B],[D,E]]))

    return (y,z)

def solver_normal(pt_1,pt_2,pt_3,pt_4):   ##put pts in y,z
    A=(pt_2[1]-pt_1[1])
    B=(pt_1[0]-pt_2[0])
    C=(-pt_2[0]*pt_1[1]+pt_2[1]*pt_1[0])

    D=(pt_4[1]-pt_3[1])
    E=(pt_3[0]-pt_4[0])
    F=(-pt_4[0]*pt_3[1]+pt_4[1]*pt_3[0])

    y=np.linalg.det(np.array([[C,B],[F,E]]))/np.linalg.det(np.array([[A,B],[D,E]]))
    z=np.linalg.det(np.array([[A,C],[D,F]]))/np.linalg.det(np.array([[A,B],[D,E]]))

    return (y,z)


c1=solver_wishbones(dict_FL['UF'],dict_FL['UA'],dict_FL['UBJ'],dict_FL['LF'],dict_FL['LA'],dict_FL['LBJ'])
c2=solver_wishbones(dict_FR['UF'],dict_FR['UA'],dict_FR['UBJ'],dict_FR['LF'],dict_FR['LA'],dict_FR['LBJ'])
res=solver_normal(c1,(dict_FL['CP'][1],dict_FL['CP'][2]),c2,(dict_FR['CP'][1],dict_FR['CP'][2]))

#solver

arr1=np.linspace(0.5,3,6)
arr2=np.linspace(-0.5,-3,6)
