import numpy as np
import pandas as pd
from hardpoint_io import dict_FL,dict_FR,dict_RL, dict_RR,FL_d1,FL_d2,FL_d3,FR_d1,FR_d2,FR_d3,RL_d1,RL_d2,RL_d3,RR_d1,RR_d2,RR_d3,FL_k_L,FL_k_U,FR_k_L,FR_k_U,RL_k_L,RL_k_U,RR_k_L,RR_k_U,rear_static_camber,rear_static_toe,front_static_camber,front_static_toe
from kinematic_solver import out_dict_FL, out_dict_FR,out_dict_RL,out_dict_RR

def line_intersection(tup1,tup2,tup3,tup4):
    alpha1=tup1[1]-tup2[1]
    beta1=tup2[0]-tup1[0]
    gamma1=(tup1[1]*tup2[0]-tup1[0]*tup2[1])

    alpha2=tup3[1]-tup4[1]
    beta2=tup4[0]-tup3[0]
    gamma2=(tup3[1]*tup4[0]-tup3[0]*tup4[1])

    mat_d=np.array([[alpha1,beta1],[alpha2,beta2]])   
    mat_dx=np.array([[gamma1,beta1],[gamma2,beta2]])
    mat_dy=np.array([[alpha1,gamma1],[alpha2,gamma2]])
    x=np.linalg.det(mat_dx)/np.linalg.det(mat_d)
    y=np.linalg.det(mat_dy)/np.linalg.det(mat_d)

    return (x,y)

#camber:
#front:
vect_front=(out_dict_FL['WC']-out_dict_FL['CP'])
camber_front=np.degrees(np.arctan2(vect_front[:,1],vect_front[:,2]))+front_static_camber
#rear
vect_rear=(out_dict_RL['WC']-out_dict_RL['CP'])
camber_rear=np.degrees(np.arctan2(vect_rear[:,1],vect_rear[:,2]))+rear_static_camber

#toe angle:
#front
vect_front=(out_dict_FL['WC']-out_dict_FL['WSP']) ##used WC, WSP instead tro,lbj becuz of some good reasons
toe_front=np.degrees(np.arctan2(vect_front[:,1],vect_front[:,0]))-90+front_static_toe
#rear
vect_rear=(out_dict_RL['WC']-out_dict_RL['WSP']) ##used WC, WSP instead tro,lbj becuz of some good reasons
toe_rear=np.degrees(np.arctan2(vect_rear[:,1],vect_rear[:,0]))-90+rear_static_toe

#caster:
#front:
vect_front=(out_dict_FL['UBJ']-out_dict_FL['LBJ'])
caster_front=np.degrees(np.arctan2(-vect_front[:,0],vect_front[:,2]))

#rear
vect_rear=(out_dict_RL['UBJ']-out_dict_RL['LBJ'])
caster_rear=np.degrees(np.arctan2(-vect_rear[:,0],vect_rear[:,2]))

#kpi
#front
vect_front=(out_dict_FL['UBJ']-out_dict_FL['LBJ'])
kpi_front=np.degrees(np.arctan2(-vect_front[:,1],vect_front[:,2]))
#rear
vect_rear=(out_dict_RL['UBJ']-out_dict_RL['LBJ'])
kpi_rear=np.degrees(np.arctan2(-vect_rear[:,1],vect_rear[:,2]))

#rc_height
def inter_pt(out_dict_corner,i,BJ,Fore,Aft):
    v1=out_dict_corner[BJ][i,:]-out_dict_corner[Fore][i,:]
    v2=out_dict_corner[Aft][i,:]-out_dict_corner[Fore][i,:]

    pointer_vect=(np.dot(v1,v2)/(np.linalg.norm(v2)))*(v2/np.linalg.norm(v2))+out_dict_corner[Fore][i,:]
    return (pointer_vect[1],pointer_vect[2])

rc_arr=np.zeros(51)
for i in range(0,51,1):
    L_upp_pointer=inter_pt(out_dict_FL,i,'UBJ','UF','UA')
    L_low_pointer=inter_pt(out_dict_FL,i,'LBJ','LF','LA')
    IC_1=line_intersection((out_dict_FL['UBJ'][i,1],out_dict_FL['UBJ'][i,2]),(L_upp_pointer[0],L_upp_pointer[1]),(out_dict_FL['LBJ'][i,1],out_dict_FL['LBJ'][i,2]),(L_low_pointer[0],L_low_pointer[1]))

    R_upp_pointer=inter_pt(out_dict_FR,i,'UBJ','UF','UA')
    R_low_pointer=inter_pt(out_dict_FR,i,'LBJ','LF','LA')
    IC_2=line_intersection((out_dict_FR['UBJ'][i,1],out_dict_FR['UBJ'][i,2]),(R_upp_pointer[0],R_upp_pointer[1]),(out_dict_FR['LBJ'][i,1],out_dict_FR['LBJ'][i,2]),(R_low_pointer[0],R_low_pointer[1]))

    rc_arr[i]=line_intersection((out_dict_FL['CP'][i,1],out_dict_FL['CP'][i,2]),(IC_1[0],IC_1[1]),(out_dict_FR['CP'][i,1],out_dict_FR['CP'][i,2]),(IC_2[0],IC_2[1]))[1]

#%anti_dive
