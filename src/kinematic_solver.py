import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import brentq
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from hardpoint_io import dict_FL, dict_FR,d1,d2,d3,k_U_Left,k_L_Left

def rodrigues_front(theta,k,BJ_stat_rel,vector_rel_origin):
    Kx=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    Kx2=((Kx)@(Kx))
    I=np.eye(3)
    R=(I+(Kx*(np.sin(theta)))+(Kx2*(1-np.cos(theta))))
    return (R@BJ_stat_rel+vector_rel_origin)                             

def upper_front(theta_U,k_U,BJ_stat_rel,vector_rel_origin):
    UBJ_curr=(rodrigues_front(theta_U,k_U,BJ_stat_rel,vector_rel_origin))
    return UBJ_curr                               

def lower_front(UBJ_curr,LBJ_stat_rel,vector_rel_origin,k_lower_side,seed):
    def residual(theta_L):
        return (np.linalg.norm(UBJ_curr-rodrigues_front(theta_L,k_lower_side,LBJ_stat_rel,vector_rel_origin))-d1)
    root=fsolve(residual,seed)[0]
    LBJ_curr=rodrigues_front(root,k_L_Left,LBJ_stat_rel,vector_rel_origin)
    return (LBJ_curr,root)



def tierod(UBJ_curr,LBJ_curr,UBJ_stat,LBJ_stat,TRO_stat,seed):
    
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

    root=fsolve(residual,seed)[0]

    TRO_curr=tro_post(root,e1,e2,radius,centre_curr)
    return (TRO_curr,root)

def triad_transform(UBJ_curr,LBJ_curr,TRO_curr,UBJ_stat,LBJ_stat,TRO_stat,vector_rel_lbj):

    e1_stat=(UBJ_stat-LBJ_stat)/(np.linalg.norm(UBJ_stat-LBJ_stat))
    v=(TRO_stat-LBJ_stat)
    e2_stat=(v-np.dot(v,e1_stat)*e1_stat)/(np.linalg.norm(v-np.dot(v,e1_stat)*e1_stat))
    e3_stat=np.cross(e1_stat,e2_stat)
    M1=np.array([e1_stat,e2_stat,e3_stat])

    e1_curr=(UBJ_curr-LBJ_curr)/(np.linalg.norm(UBJ_curr-LBJ_curr))
    v_curr=(TRO_curr-LBJ_curr)
    e2_curr=(v_curr-np.dot(v_curr,e1_curr)*e1_curr)/(np.linalg.norm(v_curr-np.dot(v_curr,e1_curr)*e1_curr))
    e3_curr=np.cross(e1_curr,e2_curr)
    M2=np.array([e1_curr,e2_curr,e3_curr])

    return (((M2.T)@(M1)@vector_rel_lbj)+LBJ_curr)  

seed1=0.0
def seeder(UBJ_stat,LBJ_stat,TRO_stat):
    
    kp_static=(UBJ_stat-LBJ_stat)/(np.linalg.norm(UBJ_stat-LBJ_stat))
    v_static=(TRO_stat-LBJ_stat)
    component=(np.dot(v_static,kp_static))
    radius=np.linalg.norm(v_static-(component*kp_static))
    
    tmp=np.array([1,0,0])
    x=(tmp-((np.dot(tmp,kp_static))/np.linalg.norm(kp_static))*kp_static)  
    e1=(x/np.linalg.norm(x))
    a=(v_static-component*kp_static)
    e2=np.cross(kp_static,e1)
    return np.arctan2((np.dot(a,e2)),(np.dot(a,e1)))

seed2=seeder(dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'])

seed3=0.0
seed4=0.0
for z in range(1,26,1):
    def WC(theta_U):
        UBJ_stat_rel=dict_FL['UBJ']-dict_FL['UA']
        UBJ_curr=upper_front(theta_U,k_U_Left,UBJ_stat_rel,dict_FL['UA'])
        LBJ_curr=lower_front(UBJ_curr,(dict_FL['LBJ']-dict_FL['LA']),dict_FL['LA'],k_L_Left,seed1)[0]
        TRO_curr=tierod(UBJ_curr,LBJ_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],seed2)[0]
        return ((triad_transform(UBJ_curr,LBJ_curr,TRO_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],(dict_FL['WC']-dict_FL['LBJ']))[2]-dict_FL['WC'][2])-z)
    try:
        root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
    except ValueError:
        try:
            root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed3} limits")
    
    seed3=root+(root-seed3)
    UBJ_stat_rel=dict_FL['UBJ']-dict_FL['UF']
    UBJ_curr=upper_front(root,k_U_Left,UBJ_stat_rel,dict_FL['UF'])

    result_LOWER=lower_front(UBJ_curr,(dict_FL['LBJ']-dict_FL['LF']),dict_FL['LF'],k_L_Left,seed1)
    seed1=(result_LOWER[1]+(result_LOWER[1]-seed1))
    LBJ_curr=result_LOWER[0]

    result_TIE=tierod(UBJ_curr,LBJ_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],seed2)
    seed2=(result_TIE[1]+(result_TIE[1]-seed2))
    TRO_curr=result_TIE[0]

    PRO_curr=rodrigues_front(root,k_U_Left,(dict_FL['PRO']-dict_FL['UA']),dict_FL['UA'])

    def residual(alpha):
        k_rocker=((dict_FL['RPA1']-dict_FL['RPA2'])/np.linalg.norm(dict_FL['RPA1']-dict_FL['RPA2']))
        return (np.linalg.norm((rodrigues_front(alpha,k_rocker,(dict_FL['PRI']-dict_FL['RPA2']),dict_FL['RPA2']))-PRO_curr)-d3)
    
    sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
    root2=sol.root
    seed4=root2+(root2-seed4)
    PRI_curr=rodrigues_front(root2,((dict_FL['RPA1']-dict_FL['RPA2'])/np.linalg.norm(dict_FL['RPA1']-dict_FL['RPA2'])),(dict_FL['PRI']-dict_FL['RPA2']),dict_FL['RPA2'])
    RD_curr=triad_transform(PRI_curr,dict_FL['RPA1'],dict_FL['RPA2'],dict_FL['PRI'],dict_FL['RPA1'],dict_FL['RPA2'],(dict_FL['RD']-dict_FL['RPA1']))

    kingpin=(UBJ_curr-LBJ_curr)
    caster=(180/np.pi)*(np.arctan2(-kingpin[0],kingpin[2]))
    print(caster)


print('\n')

seed1=0.0
seed2=seeder(dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'])
seed3=0.0
seed4=0.0
for z in range(-1,-26,-1):
    def WC(theta_U):
        UBJ_stat_rel=dict_FL['UBJ']-dict_FL['UA']
        UBJ_curr=upper_front(theta_U,k_U_Left,UBJ_stat_rel,dict_FL['UA'])
        LBJ_curr=lower_front(UBJ_curr,(dict_FL['LBJ']-dict_FL['LA']),dict_FL['LA'],k_L_Left,seed1)[0]
        TRO_curr=tierod(UBJ_curr,LBJ_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],seed2)[0]
        return ((triad_transform(UBJ_curr,LBJ_curr,TRO_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],(dict_FL['WC']-dict_FL['LBJ']))[2]-dict_FL['WC'][2])-z)
    try:
        root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
    except:
        try:
            root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed3} limits")
    
    seed3=root+(root-seed3)
    UBJ_stat_rel=dict_FL['UBJ']-dict_FL['UF']
    UBJ_curr=upper_front(root,k_U_Left,UBJ_stat_rel,dict_FL['UF'])

    result_LOWER=lower_front(UBJ_curr,(dict_FL['LBJ']-dict_FL['LF']),dict_FL['LF'],k_L_Left,seed1)
    seed1=(result_LOWER[1]+(result_LOWER[1]-seed1))
    LBJ_curr=result_LOWER[0]

    result_TIE=tierod(UBJ_curr,LBJ_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],seed2)
    seed2=(result_TIE[1]+(result_TIE[1]-seed2))
    TRO_curr=result_TIE[0]

    PRO_curr=rodrigues_front(root,k_U_Left,(dict_FL['PRO']-dict_FL['UA']),dict_FL['UA'])

    def residual(alpha):
        k_rocker=((dict_FL['RPA1']-dict_FL['RPA2'])/np.linalg.norm(dict_FL['RPA1']-dict_FL['RPA2']))
        return (np.linalg.norm((rodrigues_front(alpha,k_rocker,(dict_FL['PRI']-dict_FL['RPA2']),dict_FL['RPA2']))-PRO_curr)-d3)
    
    sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
    root2=sol.root
    seed4=root2+(root2-seed4)
    PRI_curr=rodrigues_front(root2,((dict_FL['RPA1']-dict_FL['RPA2'])/np.linalg.norm(dict_FL['RPA1']-dict_FL['RPA2'])),(dict_FL['PRI']-dict_FL['RPA2']),dict_FL['RPA2'])
    RD_curr=triad_transform(PRI_curr,dict_FL['RPA1'],dict_FL['RPA2'],dict_FL['PRI'],dict_FL['RPA1'],dict_FL['RPA2'],(dict_FL['RD']-dict_FL['RPA1']))
