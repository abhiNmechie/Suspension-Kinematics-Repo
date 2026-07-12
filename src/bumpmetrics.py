import numpy as np
import pandas as pd
from hardpoint_io import dict_FL,dict_FR,dict_RL, dict_RR,FL_d1,FL_d2,FL_d3,FR_d1,FR_d2,FR_d3,RL_d1,RL_d2,RL_d3,RR_d1,RR_d2,RR_d3,FL_k_L,FL_k_U,FR_k_L,FR_k_U,RL_k_L,RL_k_U,RR_k_L,RR_k_U,rear_static_camber,rear_static_toe,front_static_camber,front_static_camber
from kinematic_solver import out_dict_FL, out_dict_FR,out_dict_RL,out_dict_RR

#camber:
#front:
vect_front=(out_dict_FL['WC']-out_dict_FL['CP'])
camber_front=np.degrees(np.arctan2(vect_front[:,1],vect_front[:,2]))+front_static_camber
#rear
vect_rear=(out_dict_RL['WC']-out_dict_RL['CP'])
camber_rear=np.degrees(np.arctan2(vect_rear[:,1],vect_rear[:,2]))+rear_static_camber

