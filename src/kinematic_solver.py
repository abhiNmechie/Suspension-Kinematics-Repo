import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import brentq
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from hardpoint_io import dict_FL,dict_FR,dict_RL, dict_RR,FL_d1,FL_d2,FL_d3,FR_d1,FR_d2,FR_d3,RL_d1,RL_d2,RL_d3,RR_d1,RR_d2,RR_d3,FL_k_L,FL_k_U,FR_k_L,FR_k_U,RL_k_L,RL_k_U,RR_k_L,RR_k_U
np.set_printoptions(suppress=True, precision=5)

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
def corner(input_corner_dict,output_corner_dict,const_d1,const_d2,const_d3,corner_k_U,corner_k_L):
    seed1=0.0
    seed2=seeder(input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'])
    seed3=0.0
    seed4=0.0
    for z in range(1,26,1):
        index=25+z
        def WC(theta_U):
            UBJ_stat_rel=input_corner_dict['UBJ']-input_corner_dict['UA']
            UBJ_curr=upper(theta_U,corner_k_U,UBJ_stat_rel,input_corner_dict['UA'])
            LBJ_curr=lower(UBJ_curr,(input_corner_dict['LBJ']-input_corner_dict['LA']),input_corner_dict['LA'],corner_k_L,seed1,const_d1)[0]
            TRO_curr=tierod(UBJ_curr,LBJ_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],input_corner_dict['TRI'],seed2,const_d2)[0]
            return ((triad_transform(UBJ_curr,LBJ_curr,TRO_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],(input_corner_dict['WC']-input_corner_dict['LBJ']))[2]-input_corner_dict['WC'][2])-z)
        try:
            root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
        except ValueError:
            try:
                root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
            except:
                raise ValueError(f"The geometry exceeded {seed3} limits")
        
        WC_curr=WC(root)
        output_corner_dict['WC'][index]=WC_curr  #wc_curr
        seed3=root+(root-seed3)

        UBJ_stat_rel=input_corner_dict['UBJ']-input_corner_dict['UF']
        UBJ_curr=upper(root,corner_k_U,UBJ_stat_rel,input_corner_dict['UF'])  #ubj_curr exfil
        output_corner_dict['UBJ'][index]=UBJ_curr
        
        result_LOWER=lower(UBJ_curr,(input_corner_dict['LBJ']-input_corner_dict['LA']),input_corner_dict['LA'],corner_k_L,seed1,const_d1)
        seed1=(result_LOWER[1]+(result_LOWER[1]-seed1))
        LBJ_curr=result_LOWER[0]                                        #lbj_curr exfil
        output_corner_dict['LBJ'][index]=LBJ_curr 

        result_TIE=tierod(UBJ_curr,LBJ_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],input_corner_dict['TRI'],seed2,const_d2)
        seed2=(result_TIE[1]+(result_TIE[1]-seed2))
        TRO_curr=result_TIE[0]                                          #tro_curr exfil
        output_corner_dict['TRO'][index]=TRO_curr

        PRO_curr=rodrigues(root,corner_k_U,(input_corner_dict['PRO']-input_corner_dict['UA']),input_corner_dict['UA'])  #pro_curr exfil
        output_corner_dict['PRO'][index]=PRO_curr

        def residual(alpha):
            k_rocker=((input_corner_dict['RPA1']-input_corner_dict['RPA2'])/np.linalg.norm(input_corner_dict['RPA1']-input_corner_dict['RPA2']))
            return (np.linalg.norm((rodrigues(alpha,k_rocker,(input_corner_dict['PRI']-input_corner_dict['RPA2']),input_corner_dict['RPA2']))-PRO_curr)-const_d3)
        
        sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
        root2=sol.root 
        seed4=root2+(root2-seed4)
        PRI_curr=rodrigues(root2,((input_corner_dict['RPA1']-input_corner_dict['RPA2'])/np.linalg.norm(input_corner_dict['RPA1']-input_corner_dict['RPA2'])),(input_corner_dict['PRI']-input_corner_dict['RPA2']),input_corner_dict['RPA2'])  
        output_corner_dict['PRI'][index]=PRI_curr  #pri_curr

        RD_curr=triad_transform(PRI_curr,input_corner_dict['RPA1'],input_corner_dict['RPA2'],input_corner_dict['PRI'],input_corner_dict['RPA1'],input_corner_dict['RPA2'],(input_corner_dict['RD']-input_corner_dict['RPA1']))  
        output_corner_dict['RD'][index]=RD_curr  #rd_curr

        CP_curr=triad_transform(UBJ_curr,LBJ_curr,TRO_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],(input_corner_dict['CP']-input_corner_dict['LBJ'])) #cp_curr
        output_corner_dict['CP'][index]=CP_curr

        WSP_curr=triad_transform(UBJ_curr,LBJ_curr,TRO_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],(input_corner_dict['WSP']-input_corner_dict['LBJ'])) #wsp_curr
        output_corner_dict['WSP'][index]=WSP_curr
    
    seed1=0.0
    seed2=seeder(input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'])
    seed3=0.0
    seed4=0.0
    for z in range(-1,-26,-1):
        index=25+z
        def WC(theta_U):
            UBJ_stat_rel=input_corner_dict['UBJ']-input_corner_dict['UA']
            UBJ_curr=upper(theta_U,corner_k_U,UBJ_stat_rel,input_corner_dict['UA'])
            LBJ_curr=lower(UBJ_curr,(input_corner_dict['LBJ']-input_corner_dict['LA']),input_corner_dict['LA'],corner_k_L,seed1,const_d1)[0]
            TRO_curr=tierod(UBJ_curr,LBJ_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],input_corner_dict['TRI'],seed2,const_d2)[0]
            return ((triad_transform(UBJ_curr,LBJ_curr,TRO_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],(input_corner_dict['WC']-input_corner_dict['LBJ']))[2]-input_corner_dict['WC'][2])-z)
        try:
            root=brentq(WC,seed3-0.1,seed3+0.1,xtol=1e-10)
        except ValueError:
            try:
                root=brentq(WC,seed3-0.2,seed3+0.2,xtol=1e-10)
            except:
                raise ValueError(f"The geometry exceeded {seed3} limits")
        
        WC_curr=WC(root)
        output_corner_dict['WC'][index]=WC_curr  #wc_curr
        seed3=root+(root-seed3)

        UBJ_stat_rel=input_corner_dict['UBJ']-input_corner_dict['UF']
        UBJ_curr=upper(root,corner_k_U,UBJ_stat_rel,input_corner_dict['UF'])  #ubj_curr exfil
        output_corner_dict['UBJ'][index]=UBJ_curr
        
        result_LOWER=lower(UBJ_curr,(input_corner_dict['LBJ']-input_corner_dict['LA']),input_corner_dict['LA'],corner_k_L,seed1,const_d1)
        seed1=(result_LOWER[1]+(result_LOWER[1]-seed1))
        LBJ_curr=result_LOWER[0]                                        #lbj_curr exfil
        output_corner_dict['LBJ'][index]=LBJ_curr 

        result_TIE=tierod(UBJ_curr,LBJ_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],input_corner_dict['TRI'],seed2,const_d2)
        seed2=(result_TIE[1]+(result_TIE[1]-seed2))
        TRO_curr=result_TIE[0]                                          #tro_curr exfil1
        output_corner_dict['TRO'][index]=TRO_curr

        PRO_curr=rodrigues(root,corner_k_U,(input_corner_dict['PRO']-input_corner_dict['UA']),input_corner_dict['UA'])  #pro_curr exfil
        output_corner_dict['PRO'][index]=PRO_curr

        def residual(alpha):
            k_rocker=((input_corner_dict['RPA1']-input_corner_dict['RPA2'])/np.linalg.norm(input_corner_dict['RPA1']-input_corner_dict['RPA2']))
            return (np.linalg.norm((rodrigues(alpha,k_rocker,(input_corner_dict['PRI']-input_corner_dict['RPA2']),input_corner_dict['RPA2']))-PRO_curr)-const_d3)
        
        sol=root_scalar(residual,x0=seed4,x1=seed4+1e-6,method='secant')
        root2=sol.root 
        seed4=root2+(root2-seed4)
        PRI_curr=rodrigues(root2,((input_corner_dict['RPA1']-input_corner_dict['RPA2'])/np.linalg.norm(input_corner_dict['RPA1']-input_corner_dict['RPA2'])),(input_corner_dict['PRI']-input_corner_dict['RPA2']),input_corner_dict['RPA2'])  
        output_corner_dict['PRI'][index]=PRI_curr  #pri_curr

        RD_curr=triad_transform(PRI_curr,input_corner_dict['RPA1'],input_corner_dict['RPA2'],input_corner_dict['PRI'],input_corner_dict['RPA1'],input_corner_dict['RPA2'],(input_corner_dict['RD']-input_corner_dict['RPA1']))  
        output_corner_dict['RD'][index]=RD_curr  #rd_curr

        CP_curr=triad_transform(UBJ_curr,LBJ_curr,TRO_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],(input_corner_dict['CP']-input_corner_dict['LBJ'])) #cp_curr
        output_corner_dict['CP'][index]=CP_curr

        WSP_curr=triad_transform(UBJ_curr,LBJ_curr,TRO_curr,input_corner_dict['UBJ'],input_corner_dict['LBJ'],input_corner_dict['TRO'],(input_corner_dict['WSP']-input_corner_dict['LBJ'])) #wsp_curr
        output_corner_dict['WSP'][index]=WSP_curr

    #completely static vals:
    output_corner_dict['UA'][:]=input_corner_dict['UA']
    output_corner_dict['UF'][:]=input_corner_dict['UF']
    output_corner_dict['LA'][:]=input_corner_dict['LA']
    output_corner_dict['LF'][:]=input_corner_dict['LF']
    output_corner_dict['RPA1'][:]=input_corner_dict['RPA1']
    output_corner_dict['RPA2'][:]=input_corner_dict['RPA2']
    output_corner_dict['DC'][:]=input_corner_dict['DC']
    output_corner_dict['TRI'][:]=input_corner_dict['TRI']

corner(dict_FL,out_dict_FL,FL_d1,FL_d2,FL_d3,FL_k_U,FL_k_L)
corner(dict_FR,out_dict_FR,FR_d1,FR_d2,FR_d3,FR_k_U,FR_k_L)
corner(dict_RL,out_dict_RL,RL_d1,RL_d2,RL_d3,RL_k_U,RL_k_L)
corner(dict_RR,out_dict_RR,RR_d1,RR_d2,RR_d3,RR_k_U,RR_k_L)

kp=(out_dict_FL['UBJ']-out_dict_FL['LBJ'])