import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import brentq
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from hardpoint_io import dict_FL,dict_FR,dict_RL, dict_RR,FL_d1,FL_d2,FL_d3,FR_d1,FR_d2,FR_d3,RL_d1,RL_d2,RL_d3,RR_d1,RR_d2,RR_d3,FL_k_L,FL_k_U,FR_k_L,FR_k_U,RL_k_L,RL_k_U,RR_k_L,RR_k_U

#output dictionary containing array:
#for front ledft:
out_dict_FL=dict()
for k,v in dict_FL.items():
    out_dict_FL[k]=np.zeros((51,3))
    out_dict_FL[k][25]=v

#for front right:
out_dict_FR=dict()
for k,v in dict_FR.items():
    out_dict_FR[k]=np.zeros((51,3))
    out_dict_FR[k][25]=v

#for rear left  :
out_dict_RL=dict()
for k,v in dict_RL.items():
    out_dict_RL[k]=np.zeros((51,3))
    out_dict_RL[k][25]=v

#for rear right:
out_dict_RR=dict()
for k,v in dict_RR.items():
    out_dict_RR[k]=np.zeros((51,3))
    out_dict_RR[k][25]=v


#BUMP SOLVER:
def rodrigues(theta,k,BJ_stat_rel,vector_rel_origin):
    Kx=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    Kx2=((Kx)@(Kx))
    I=np.eye(3)
    R=(I+(Kx*(np.sin(theta)))+(Kx2*(1-np.cos(theta))))
    return (R@BJ_stat_rel+vector_rel_origin)                             

def upper(theta_U,k_U,BJ_stat_rel,vector_rel_origin):
    UBJ_curr=(rodrigues(theta_U,k_U,BJ_stat_rel,vector_rel_origin))
    return UBJ_curr                               

def lower(UBJ_curr,LBJ_stat_rel,vector_rel_origin,k_lower_side,seed,const_dist):
    def residual(theta_L):
        return (np.linalg.norm(UBJ_curr-rodrigues(theta_L,k_lower_side,LBJ_stat_rel,vector_rel_origin))-const_dist)
    root=fsolve(residual,seed)[0]
    LBJ_curr=rodrigues(root,k_lower_side,LBJ_stat_rel,vector_rel_origin)
    return (LBJ_curr,root)

def tierod(UBJ_curr,LBJ_curr,UBJ_stat,LBJ_stat,TRO_stat,TRI_stat,seed,const_dist):
    
    kp_static=(UBJ_stat-LBJ_stat)/(np.linalg.norm(UBJ_stat-LBJ_stat))
    v_static=(TRO_stat-LBJ_stat)
    component=(np.dot(v_static,kp_static))
    radius=np.linalg.norm(v_static-(component*kp_static))
    
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
        return (np.linalg.norm(tro_post(phi,e1,e2,radius,centre_curr)-TRI_stat)-const_dist)

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

# front left
seed1=0.0
seed2=seeder(dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'])
seed3=0.0
seed4=0.0
for z in range(1,26,1):
    def WC(theta_U):
        UBJ_stat_rel=dict_FL['UBJ']-dict_FL['UA']
        FL_UBJ_curr=upper(theta_U,FL_k_U,UBJ_stat_rel,dict_FL['UA'])
        FL_LBJ_curr=lower(FL_UBJ_curr,(dict_FL['LBJ']-dict_FL['LA']),dict_FL['LA'],FL_k_L,seed1,FL_d1)[0]
        FL_TRO_curr=tierod(FL_UBJ_curr,FL_LBJ_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],dict_FL['TRI'],seed2,FL_d2)[0]
        return ((triad_transform(FL_UBJ_curr,FL_LBJ_curr,FL_TRO_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],(dict_FL['WC']-dict_FL['LBJ']))[2]-dict_FL['WC'][2])-z)
    try:
        root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
    except ValueError:
        try:
            root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed3} limits")
    
    seed3=root+(root-seed3)
    FL_UBJ_stat_rel=dict_FL['UBJ']-dict_FL['UF']
    FL_UBJ_curr=upper(root,FL_k_U,FL_UBJ_stat_rel,dict_FL['UF'])  #ubj_curr exfil

    FL_result_LOWER=lower(FL_UBJ_curr,(dict_FL['LBJ']-dict_FL['LA']),dict_FL['LA'],FL_k_L,seed1,FL_d1)
    seed1=(FL_result_LOWER[1]+(FL_result_LOWER[1]-seed1))
    FL_LBJ_curr=FL_result_LOWER[0]                                        #lbj_curr exfil

    FL_result_TIE=tierod(FL_UBJ_curr,FL_LBJ_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],dict_FL['TRI'],seed2,FL_d2)
    seed2=(FL_result_TIE[1]+(FL_result_TIE[1]-seed2))
    FL_TRO_curr=FL_result_TIE[0]                                          #tro_curr exfil

    FL_PRO_curr=rodrigues(root,FL_k_U,(dict_FL['PRO']-dict_FL['UA']),dict_FL['UA'])  #pro_curr exfil

    def residual(alpha):
        k_rocker=((dict_FL['RPA1']-dict_FL['RPA2'])/np.linalg.norm(dict_FL['RPA1']-dict_FL['RPA2']))
        return (np.linalg.norm((rodrigues(alpha,k_rocker,(dict_FL['PRI']-dict_FL['RPA2']),dict_FL['RPA2']))-FL_PRO_curr)-FL_d3)
    
    sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
    root2=sol.root 
    seed4=root2+(root2-seed4)
    FL_PRI_curr=rodrigues(root2,((dict_FL['RPA1']-dict_FL['RPA2'])/np.linalg.norm(dict_FL['RPA1']-dict_FL['RPA2'])),(dict_FL['PRI']-dict_FL['RPA2']),dict_FL['RPA2'])  #pri_curr
    FL_RD_curr=triad_transform(FL_PRI_curr,dict_FL['RPA1'],dict_FL['RPA2'],dict_FL['PRI'],dict_FL['RPA1'],dict_FL['RPA2'],(dict_FL['RD']-dict_FL['RPA1']))   ##rd_cur

    kingpin=(FL_UBJ_curr-FL_LBJ_curr)
    caster=(180/np.pi)*(np.arctan2(-kingpin[0],kingpin[2]))


print('\n')

seed1=0.0
seed2=seeder(dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'])
seed3=0.0
seed4=0.0
for z in range(-1,-26,-1):
    def WC(theta_U):
        UBJ_stat_rel=dict_FL['UBJ']-dict_FL['UA']
        FL_UBJ_curr=upper(theta_U,FL_k_U,UBJ_stat_rel,dict_FL['UA'])
        FL_LBJ_curr=lower(FL_UBJ_curr,(dict_FL['LBJ']-dict_FL['LA']),dict_FL['LA'],FL_k_L,seed1,FL_d1)[0]
        FL_TRO_curr=tierod(FL_UBJ_curr,FL_LBJ_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],dict_FL['TRI'],seed2,FL_d2)[0]
        return ((triad_transform(FL_UBJ_curr,FL_LBJ_curr,FL_TRO_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],(dict_FL['WC']-dict_FL['LBJ']))[2]-dict_FL['WC'][2])-z)
    try:
        root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
    except ValueError:
        try:
            root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed3} limits")
    
    seed3=root+(root-seed3)
    FL_UBJ_stat_rel=dict_FL['UBJ']-dict_FL['UF']
    FL_UBJ_curr=upper(root,FL_k_U,FL_UBJ_stat_rel,dict_FL['UF'])  #ubj_curr exfil

    FL_result_LOWER=lower(FL_UBJ_curr,(dict_FL['LBJ']-dict_FL['LA']),dict_FL['LA'],FL_k_L,seed1,FL_d1)
    seed1=(FL_result_LOWER[1]+(FL_result_LOWER[1]-seed1))
    FL_LBJ_curr=FL_result_LOWER[0]                                        #lbj_curr exfil

    FL_result_TIE=tierod(FL_UBJ_curr,FL_LBJ_curr,dict_FL['UBJ'],dict_FL['LBJ'],dict_FL['TRO'],dict_FL['TRI'],seed2,FL_d2)
    seed2=(FL_result_TIE[1]+(FL_result_TIE[1]-seed2))
    FL_TRO_curr=FL_result_TIE[0]                                          #tro_curr exfil

    FL_PRO_curr=rodrigues(root,FL_k_U,(dict_FL['PRO']-dict_FL['UA']),dict_FL['UA'])  #pro_curr exfil

    def residual(alpha):
        k_rocker=((dict_FL['RPA1']-dict_FL['RPA2'])/np.linalg.norm(dict_FL['RPA1']-dict_FL['RPA2']))
        return (np.linalg.norm((rodrigues(alpha,k_rocker,(dict_FL['PRI']-dict_FL['RPA2']),dict_FL['RPA2']))-FL_PRO_curr)-FL_d3)
    
    sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
    root2=sol.root 
    seed4=root2+(root2-seed4)
    FL_PRI_curr=rodrigues(root2,((dict_FL['RPA1']-dict_FL['RPA2'])/np.linalg.norm(dict_FL['RPA1']-dict_FL['RPA2'])),(dict_FL['PRI']-dict_FL['RPA2']),dict_FL['RPA2'])  #pri_curr
    FL_RD_curr=triad_transform(FL_PRI_curr,dict_FL['RPA1'],dict_FL['RPA2'],dict_FL['PRI'],dict_FL['RPA1'],dict_FL['RPA2'],(dict_FL['RD']-dict_FL['RPA1']))   ##rd_cur

    kingpin=(FL_UBJ_curr-FL_LBJ_curr)
    caster=(180/np.pi)*(np.arctan2(-kingpin[0],kingpin[2]))

print('\n')
# front right
seed1=0.0
seed2=seeder(dict_FR['UBJ'],dict_FR['LBJ'],dict_FR['TRO'])
seed3=0.0
seed4=0.0
for z in range(1,26,1):
    def WC(theta_U):
        UBJ_stat_rel=dict_FR['UBJ']-dict_FR['UA']
        FR_UBJ_curr=upper(theta_U,FR_k_U,UBJ_stat_rel,dict_FR['UA'])
        FR_LBJ_curr=lower(FR_UBJ_curr,(dict_FR['LBJ']-dict_FR['LA']),dict_FR['LA'],FR_k_L,seed1,FR_d1)[0]
        FR_TRO_curr=tierod(FR_UBJ_curr,FR_LBJ_curr,dict_FR['UBJ'],dict_FR['LBJ'],dict_FR['TRO'],dict_FR['TRI'],seed2,FR_d2)[0]
        return ((triad_transform(FR_UBJ_curr,FR_LBJ_curr,FR_TRO_curr,dict_FR['UBJ'],dict_FR['LBJ'],dict_FR['TRO'],(dict_FR['WC']-dict_FR['LBJ']))[2]-dict_FR['WC'][2])-z)
    try:
        root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
    except ValueError:
        try:
            root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed3} limits")
    
    seed3=root+(root-seed3)
    FR_UBJ_stat_rel=dict_FR['UBJ']-dict_FR['UF']
    FR_UBJ_curr=upper(root,FR_k_U,FR_UBJ_stat_rel,dict_FR['UF'])  #ubj_curr exfil

    FR_result_LOWER=lower(FR_UBJ_curr,(dict_FR['LBJ']-dict_FR['LA']),dict_FR['LA'],FR_k_L,seed1,FR_d1)
    seed1=(FR_result_LOWER[1]+(FR_result_LOWER[1]-seed1))
    FR_LBJ_curr=FR_result_LOWER[0]                                        #lbj_curr exfil

    FR_result_TIE=tierod(FR_UBJ_curr,FR_LBJ_curr,dict_FR['UBJ'],dict_FR['LBJ'],dict_FR['TRO'],dict_FR['TRI'],seed2,FR_d2)
    seed2=(FR_result_TIE[1]+(FR_result_TIE[1]-seed2))
    FR_TRO_curr=FR_result_TIE[0]                                          #tro_curr exfil

    FR_PRO_curr=rodrigues(root,FR_k_U,(dict_FR['PRO']-dict_FR['UA']),dict_FR['UA'])  #pro_curr exfil

    def residual(alpha):
        k_rocker=((dict_FR['RPA1']-dict_FR['RPA2'])/np.linalg.norm(dict_FR['RPA1']-dict_FR['RPA2']))
        return (np.linalg.norm((rodrigues(alpha,k_rocker,(dict_FR['PRI']-dict_FR['RPA2']),dict_FR['RPA2']))-FR_PRO_curr)-FR_d3)
    
    sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
    root2=sol.root 
    seed4=root2+(root2-seed4)
    FR_PRI_curr=rodrigues(root2,((dict_FR['RPA1']-dict_FR['RPA2'])/np.linalg.norm(dict_FR['RPA1']-dict_FR['RPA2'])),(dict_FR['PRI']-dict_FR['RPA2']),dict_FR['RPA2'])  #pri_curr
    FR_RD_curr=triad_transform(FR_PRI_curr,dict_FR['RPA1'],dict_FR['RPA2'],dict_FR['PRI'],dict_FR['RPA1'],dict_FR['RPA2'],(dict_FR['RD']-dict_FR['RPA1']))   ##rd_cur

    kingpin=(FR_UBJ_curr-FR_LBJ_curr)
    caster=(180/np.pi)*(np.arctan2(-kingpin[0],kingpin[2]))

print('\n')

seed1=0.0
seed2=seeder(dict_FR['UBJ'],dict_FR['LBJ'],dict_FR['TRO'])
seed3=0.0
seed4=0.0
for z in range(-1,-26,-1):
    def WC(theta_U):
        UBJ_stat_rel=dict_FR['UBJ']-dict_FR['UA']
        FR_UBJ_curr=upper(theta_U,FR_k_U,UBJ_stat_rel,dict_FR['UA'])
        FR_LBJ_curr=lower(FR_UBJ_curr,(dict_FR['LBJ']-dict_FR['LA']),dict_FR['LA'],FR_k_L,seed1,FR_d1)[0]
        FR_TRO_curr=tierod(FR_UBJ_curr,FR_LBJ_curr,dict_FR['UBJ'],dict_FR['LBJ'],dict_FR['TRO'],dict_FR['TRI'],seed2,FR_d2)[0]
        return ((triad_transform(FR_UBJ_curr,FR_LBJ_curr,FR_TRO_curr,dict_FR['UBJ'],dict_FR['LBJ'],dict_FR['TRO'],(dict_FR['WC']-dict_FR['LBJ']))[2]-dict_FR['WC'][2])-z)
    try:
        root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
    except ValueError:
        try:
            root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed3} limits")
    
    seed3=root+(root-seed3)
    FR_UBJ_stat_rel=dict_FR['UBJ']-dict_FR['UF']
    FR_UBJ_curr=upper(root,FR_k_U,FR_UBJ_stat_rel,dict_FR['UF'])  #ubj_curr exfil

    FR_result_LOWER=lower(FR_UBJ_curr,(dict_FR['LBJ']-dict_FR['LA']),dict_FR['LA'],FR_k_L,seed1,FR_d1)
    seed1=(FR_result_LOWER[1]+(FR_result_LOWER[1]-seed1))
    FR_LBJ_curr=FR_result_LOWER[0]                                        #lbj_curr exfil

    FR_result_TIE=tierod(FR_UBJ_curr,FR_LBJ_curr,dict_FR['UBJ'],dict_FR['LBJ'],dict_FR['TRO'],dict_FR['TRI'],seed2,FR_d2)
    seed2=(FR_result_TIE[1]+(FR_result_TIE[1]-seed2))
    FR_TRO_curr=FR_result_TIE[0]                                          #tro_curr exfil

    FR_PRO_curr=rodrigues(root,FR_k_U,(dict_FR['PRO']-dict_FR['UA']),dict_FR['UA'])  #pro_curr exfil

    def residual(alpha):
        k_rocker=((dict_FR['RPA1']-dict_FR['RPA2'])/np.linalg.norm(dict_FR['RPA1']-dict_FR['RPA2']))
        return (np.linalg.norm((rodrigues(alpha,k_rocker,(dict_FR['PRI']-dict_FR['RPA2']),dict_FR['RPA2']))-FR_PRO_curr)-FR_d3)
    
    sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
    root2=sol.root 
    seed4=root2+(root2-seed4)
    FR_PRI_curr=rodrigues(root2,((dict_FR['RPA1']-dict_FR['RPA2'])/np.linalg.norm(dict_FR['RPA1']-dict_FR['RPA2'])),(dict_FR['PRI']-dict_FR['RPA2']),dict_FR['RPA2'])  #pri_curr
    FR_RD_curr=triad_transform(FR_PRI_curr,dict_FR['RPA1'],dict_FR['RPA2'],dict_FR['PRI'],dict_FR['RPA1'],dict_FR['RPA2'],(dict_FR['RD']-dict_FR['RPA1']))   ##rd_cur

    kingpin=(FR_UBJ_curr-FR_LBJ_curr)
    caster=(180/np.pi)*(np.arctan2(-kingpin[0],kingpin[2]))


# rear left 
seed1=0.0
seed2=seeder(dict_RL['UBJ'],dict_RL['LBJ'],dict_RL['TRO'])
seed3=0.0
seed4=0.0
for z in range(1,26,1):
    def WC(theta_U):
        UBJ_stat_rel=dict_RL['UBJ']-dict_RL['UA']
        RL_UBJ_curr=upper(theta_U,RL_k_U,UBJ_stat_rel,dict_RL['UA'])
        RL_LBJ_curr=lower(RL_UBJ_curr,(dict_RL['LBJ']-dict_RL['LA']),dict_RL['LA'],RL_k_L,seed1,RL_d1)[0]
        RL_TRO_curr=tierod(RL_UBJ_curr,RL_LBJ_curr,dict_RL['UBJ'],dict_RL['LBJ'],dict_RL['TRO'],dict_RL['TRI'],seed2,RL_d2)[0]
        return ((triad_transform(RL_UBJ_curr,RL_LBJ_curr,RL_TRO_curr,dict_RL['UBJ'],dict_RL['LBJ'],dict_RL['TRO'],(dict_RL['WC']-dict_RL['LBJ']))[2]-dict_RL['WC'][2])-z)
    try:
        root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
    except ValueError:
        try:
            root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed3} limits")
    
    seed3=root+(root-seed3)
    RL_UBJ_stat_rel=dict_RL['UBJ']-dict_RL['UF']
    RL_UBJ_curr=upper(root,RL_k_U,RL_UBJ_stat_rel,dict_RL['UF'])  

    RL_result_LOWER=lower(RL_UBJ_curr,(dict_RL['LBJ']-dict_RL['LA']),dict_RL['LA'],RL_k_L,seed1,RL_d1)
    seed1=(RL_result_LOWER[1]+(RL_result_LOWER[1]-seed1))
    RL_LBJ_curr=RL_result_LOWER[0]                                       

    RL_result_TIE=tierod(RL_UBJ_curr,RL_LBJ_curr,dict_RL['UBJ'],dict_RL['LBJ'],dict_RL['TRO'],dict_RL['TRI'],seed2,RL_d2)
    seed2=(RL_result_TIE[1]+(RL_result_TIE[1]-seed2))
    RL_TRO_curr=RL_result_TIE[0]                                    

    RL_PRO_curr=rodrigues(root,RL_k_U,(dict_RL['PRO']-dict_RL['UA']),dict_RL['UA'])

    def residual(alpha):
        k_rocker=((dict_RL['RPA1']-dict_RL['RPA2'])/np.linalg.norm(dict_RL['RPA1']-dict_RL['RPA2']))
        return (np.linalg.norm((rodrigues(alpha,k_rocker,(dict_RL['PRI']-dict_RL['RPA2']),dict_RL['RPA2']))-RL_PRO_curr)-RL_d3)
    
    sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
    root2=sol.root 
    seed4=root2+(root2-seed4)
    RL_PRI_curr=rodrigues(root2,((dict_RL['RPA1']-dict_RL['RPA2'])/np.linalg.norm(dict_RL['RPA1']-dict_RL['RPA2'])),(dict_RL['PRI']-dict_RL['RPA2']),dict_RL['RPA2'])  #pri_curr
    RL_RD_curr=triad_transform(RL_PRI_curr,dict_RL['RPA1'],dict_RL['RPA2'],dict_RL['PRI'],dict_RL['RPA1'],dict_RL['RPA2'],(dict_RL['RD']-dict_RL['RPA1']))   ##rd_cur

    kingpin=(RL_UBJ_curr-RL_LBJ_curr)
    caster=(180/np.pi)*(np.arctan2(-kingpin[0],kingpin[2]))
    print(caster)

print('\n')

seed1=0.0
seed2=seeder(dict_RL['UBJ'],dict_RL['LBJ'],dict_RL['TRO'])
seed3=0.0
seed4=0.0
for z in range(-1,-26,-1):
    def WC(theta_U):
        UBJ_stat_rel=dict_RL['UBJ']-dict_RL['UA']
        RL_UBJ_curr=upper(theta_U,RL_k_U,UBJ_stat_rel,dict_RL['UA'])
        RL_LBJ_curr=lower(RL_UBJ_curr,(dict_RL['LBJ']-dict_RL['LA']),dict_RL['LA'],RL_k_L,seed1,RL_d1)[0]
        RL_TRO_curr=tierod(RL_UBJ_curr,RL_LBJ_curr,dict_RL['UBJ'],dict_RL['LBJ'],dict_RL['TRO'],dict_RL['TRI'],seed2,RL_d2)[0]
        return ((triad_transform(RL_UBJ_curr,RL_LBJ_curr,RL_TRO_curr,dict_RL['UBJ'],dict_RL['LBJ'],dict_RL['TRO'],(dict_RL['WC']-dict_RL['LBJ']))[2]-dict_RL['WC'][2])-z)
    try:
        root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
    except ValueError:
        try:
            root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed3} limits")
    
    seed3=root+(root-seed3)
    RL_UBJ_stat_rel=dict_RL['UBJ']-dict_RL['UF']
    RL_UBJ_curr=upper(root,RL_k_U,RL_UBJ_stat_rel,dict_RL['UF'])  #ubj_curr exfil

    RL_result_LOWER=lower(RL_UBJ_curr,(dict_RL['LBJ']-dict_RL['LA']),dict_RL['LA'],RL_k_L,seed1,RL_d1)
    seed1=(RL_result_LOWER[1]+(RL_result_LOWER[1]-seed1))
    RL_LBJ_curr=RL_result_LOWER[0]                                        #lbj_curr exfil

    RL_result_TIE=tierod(RL_UBJ_curr,RL_LBJ_curr,dict_RL['UBJ'],dict_RL['LBJ'],dict_RL['TRO'],dict_RL['TRI'],seed2,RL_d2)
    seed2=(RL_result_TIE[1]+(RL_result_TIE[1]-seed2))
    RL_TRO_curr=RL_result_TIE[0]                                          #tro_curr exfil

    RL_PRO_curr=rodrigues(root,RL_k_U,(dict_RL['PRO']-dict_RL['UA']),dict_RL['UA'])  #pro_curr exfil

    def residual(alpha):
        k_rocker=((dict_RL['RPA1']-dict_RL['RPA2'])/np.linalg.norm(dict_RL['RPA1']-dict_RL['RPA2']))
        return (np.linalg.norm((rodrigues(alpha,k_rocker,(dict_RL['PRI']-dict_RL['RPA2']),dict_RL['RPA2']))-RL_PRO_curr)-RL_d3)
    
    sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
    root2=sol.root 
    seed4=root2+(root2-seed4)
    RL_PRI_curr=rodrigues(root2,((dict_RL['RPA1']-dict_RL['RPA2'])/np.linalg.norm(dict_RL['RPA1']-dict_RL['RPA2'])),(dict_RL['PRI']-dict_RL['RPA2']),dict_RL['RPA2'])  #pri_curr
    RL_RD_curr=triad_transform(RL_PRI_curr,dict_RL['RPA1'],dict_RL['RPA2'],dict_RL['PRI'],dict_RL['RPA1'],dict_RL['RPA2'],(dict_RL['RD']-dict_RL['RPA1']))   ##rd_cur

    kingpin=(RL_UBJ_curr-RL_LBJ_curr)
    caster=(180/np.pi)*(np.arctan2(-kingpin[0],kingpin[2]))
    print(caster)

print('\n')

#rear right
seed1=0.0
seed2=seeder(dict_RR['UBJ'],dict_RR['LBJ'],dict_RR['TRO'])
seed3=0.0
seed4=0.0
for z in range(1,26,1):
    def WC(theta_U):
        UBJ_stat_rel=dict_RR['UBJ']-dict_RR['UA']
        RR_UBJ_curr=upper(theta_U,RR_k_U,UBJ_stat_rel,dict_RR['UA'])
        RR_LBJ_curr=lower(RR_UBJ_curr,(dict_RR['LBJ']-dict_RR['LA']),dict_RR['LA'],RR_k_L,seed1,RR_d1)[0]
        RR_TRO_curr=tierod(RR_UBJ_curr,RR_LBJ_curr,dict_RR['UBJ'],dict_RR['LBJ'],dict_RR['TRO'],dict_RR['TRI'],seed2,RR_d2)[0]
        return ((triad_transform(RR_UBJ_curr,RR_LBJ_curr,RR_TRO_curr,dict_RR['UBJ'],dict_RR['LBJ'],dict_RR['TRO'],(dict_RR['WC']-dict_RR['LBJ']))[2]-dict_RR['WC'][2])-z)
    try:
        root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
    except ValueError:
        try:
            root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed3} limits")
    
    seed3=root+(root-seed3)
    RR_UBJ_stat_rel=dict_RR['UBJ']-dict_RR['UF']
    RR_UBJ_curr=upper(root,RR_k_U,RR_UBJ_stat_rel,dict_RR['UF'])  #ubj_curr exfil

    RR_result_LOWER=lower(RR_UBJ_curr,(dict_RR['LBJ']-dict_RR['LA']),dict_RR['LA'],RR_k_L,seed1,RR_d1)
    seed1=(RR_result_LOWER[1]+(RR_result_LOWER[1]-seed1))
    RR_LBJ_curr=RR_result_LOWER[0]                                        #lbj_curr exfil

    RR_result_TIE=tierod(RR_UBJ_curr,RR_LBJ_curr,dict_RR['UBJ'],dict_RR['LBJ'],dict_RR['TRO'],dict_RR['TRI'],seed2,RR_d2)
    seed2=(RR_result_TIE[1]+(RR_result_TIE[1]-seed2))
    RR_TRO_curr=RR_result_TIE[0]                                          #tro_curr exfil

    RR_PRO_curr=rodrigues(root,RR_k_U,(dict_RR['PRO']-dict_RR['UA']),dict_RR['UA'])  #pro_curr exfil

    def residual(alpha):
        k_rocker=((dict_RR['RPA1']-dict_RR['RPA2'])/np.linalg.norm(dict_RR['RPA1']-dict_RR['RPA2']))
        return (np.linalg.norm((rodrigues(alpha,k_rocker,(dict_RR['PRI']-dict_RR['RPA2']),dict_RR['RPA2']))-RR_PRO_curr)-RR_d3)
    
    sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
    root2=sol.root 
    seed4=root2+(root2-seed4)
    RR_PRI_curr=rodrigues(root2,((dict_RR['RPA1']-dict_RR['RPA2'])/np.linalg.norm(dict_RR['RPA1']-dict_RR['RPA2'])),(dict_RR['PRI']-dict_RR['RPA2']),dict_RR['RPA2'])  #pri_curr
    RR_RD_curr=triad_transform(RR_PRI_curr,dict_RR['RPA1'],dict_RR['RPA2'],dict_RR['PRI'],dict_RR['RPA1'],dict_RR['RPA2'],(dict_RR['RD']-dict_RR['RPA1']))   ##rd_cur

    kingpin=(RR_UBJ_curr-RR_LBJ_curr)
    caster=(180/np.pi)*(np.arctan2(-kingpin[0],kingpin[2]))
    print(caster)

print('\n')

seed1=0.0
seed2=seeder(dict_RR['UBJ'],dict_RR['LBJ'],dict_RR['TRO'])
seed3=0.0
seed4=0.0
for z in range(-1,-26,-1):
    def WC(theta_U):
        UBJ_stat_rel=dict_RR['UBJ']-dict_RR['UA']
        RR_UBJ_curr=upper(theta_U,RR_k_U,UBJ_stat_rel,dict_RR['UA'])
        RR_LBJ_curr=lower(RR_UBJ_curr,(dict_RR['LBJ']-dict_RR['LA']),dict_RR['LA'],RR_k_L,seed1,RR_d1)[0]
        RR_TRO_curr=tierod(RR_UBJ_curr,RR_LBJ_curr,dict_RR['UBJ'],dict_RR['LBJ'],dict_RR['TRO'],dict_RR['TRI'],seed2,RR_d2)[0]
        return ((triad_transform(RR_UBJ_curr,RR_LBJ_curr,RR_TRO_curr,dict_RR['UBJ'],dict_RR['LBJ'],dict_RR['TRO'],(dict_RR['WC']-dict_RR['LBJ']))[2]-dict_RR['WC'][2])-z)
    try:
        root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
    except ValueError:
        try:
            root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
        except:
            raise ValueError(f"The geometry exceeded {seed3} limits")
    
    seed3=root+(root-seed3)
    RR_UBJ_stat_rel=dict_RR['UBJ']-dict_RR['UF']
    RR_UBJ_curr=upper(root,RR_k_U,RR_UBJ_stat_rel,dict_RR['UF'])  #ubj_curr exfil ho rha h 

    RR_result_LOWER=lower(RR_UBJ_curr,(dict_RR['LBJ']-dict_RR['LA']),dict_RR['LA'],RR_k_L,seed1,RR_d1)
    seed1=(RR_result_LOWER[1]+(RR_result_LOWER[1]-seed1))
    RR_LBJ_curr=RR_result_LOWER[0]                                        #lbj_curr exfil

    RR_result_TIE=tierod(RR_UBJ_curr,RR_LBJ_curr,dict_RR['UBJ'],dict_RR['LBJ'],dict_RR['TRO'],dict_RR['TRI'],seed2,RR_d2)
    seed2=(RR_result_TIE[1]+(RR_result_TIE[1]-seed2))
    RR_TRO_curr=RR_result_TIE[0]                                          #tro_curr exfil

    RR_PRO_curr=rodrigues(root,RR_k_U,(dict_RR['PRO']-dict_RR['UA']),dict_RR['UA'])  #pro_curr exfil

    def residual(alpha):
        k_rocker=((dict_RR['RPA1']-dict_RR['RPA2'])/np.linalg.norm(dict_RR['RPA1']-dict_RR['RPA2']))
        return (np.linalg.norm((rodrigues(alpha,k_rocker,(dict_RR['PRI']-dict_RR['RPA2']),dict_RR['RPA2']))-RR_PRO_curr)-RR_d3)
    
    sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
    root2=sol.root 
    seed4=root2+(root2-seed4)
    RR_PRI_curr=rodrigues(root2,((dict_RR['RPA1']-dict_RR['RPA2'])/np.linalg.norm(dict_RR['RPA1']-dict_RR['RPA2'])),(dict_RR['PRI']-dict_RR['RPA2']),dict_RR['RPA2'])  #pri_curr
    RR_RD_curr=triad_transform(RR_PRI_curr,dict_RR['RPA1'],dict_RR['RPA2'],dict_RR['PRI'],dict_RR['RPA1'],dict_RR['RPA2'],(dict_RR['RD']-dict_RR['RPA1']))   ##rd_cur

    kingpin=(RR_UBJ_curr-RR_LBJ_curr)
    caster=(180/np.pi)*(np.arctan2(-kingpin[0],kingpin[2]))
    print(caster)