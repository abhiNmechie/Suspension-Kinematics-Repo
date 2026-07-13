#SYMBOL:
# 1)UA: UPPER WISHBONE AFT (WISHBONE CHASSIS)
# 2)UF: UPPER WISHBONE FORE (WISHBONE CHASSIS)
# 3)UBJ: UPPER WISHBONE BALL JOINT (WISHBONE UPRIGHT)
# 4)LA: LOWER WISHBONE AFT (WISHBONE CHASSIS)
# 5)LF: LOWER WISHBONE FORE (WISHBONE CHASSIS)
# 6)LBJ: LOWER WISHBONE BALL JOINT (WISHBONE UPRIGHT)
# 7)TRI: TIE ROD INBOARD POINT (TIE ROD STEERING RACK)
# 8)TRO: TIE ROD OUTBOARD POINT (TIE ROD UPRIGHT)
# 9)WC: WHEEL CENTRE (UPRIGHT: RIGIDLY OFFSET)
# 10)CP: CONTACT PATCH(DIRECTLY BELOW WHEEL CENTRE, OFFSET BY LOADED TIRE RADIUS)
# 11)PRO: PUSHROD OUTBOARD PICKUP POINT (ON UPPER WISHBONE)
# 12)RPA1: ROCKER PIVOT AXIS POINT 1
# 13)RPA2: ROCKER PIVOT AXIS POINT 2
# 14)RPR: ROCKER-PUSHROD POINT
# 15)RD: ROCKER-DAMPER POINT
# 16)DC: DAMPER-CHASSIS POINT

import numpy as np
import pandas as pd

np.set_printoptions(suppress=True, precision=3)

#FRONT SUSPENSION HARDPOINTS
frontsus_df=pd.read_csv(r"C:\PROGRAMMING\Codes\SUSPENSION KINEMATICS SIMULATOR\CODE REPOSITORY\data\frontsusFINAL2.txt")
frontsus_df.set_index('LABEL',inplace=True)

dict_FL=dict()
dict_FL={
    'LF':np.empty(3),
    'LA':np.empty(3),
    'LBJ':np.empty(3),
    'UF':np.empty(3),
    'UA':np.empty(3),
    'UBJ':np.empty(3),
    'PRO':np.empty(3),
    'PRI':np.empty(3),
    'TRO':np.empty(3),
    'TRI':np.empty(3),
    'DC':np.empty(3),
    'RD':np.empty(3),
    'WC':np.empty(3),
    'RPA1':np.empty(3),
    'RPA2':np.empty(3),
    'CP':np.empty(3),
    'WSP':np.empty(3)
}
for k,v in dict_FL.items():
    temp=frontsus_df.loc[k,['X','Y','Z']].values
    temp=(np.array([[-1,0,0],[0,1,0],[0,0,1]])@temp)
    temp[0]=temp[0]+1000
    dict_FL[k]=temp

dict_FR=dict()
dict_FR={
    'LF':np.empty(3),
    'LA':np.empty(3),
    'LBJ':np.empty(3),
    'UF':np.empty(3),
    'UA':np.empty(3),
    'UBJ':np.empty(3),
    'PRO':np.empty(3),
    'PRI':np.empty(3),
    'TRO':np.empty(3),
    'TRI':np.empty(3),
    'DC':np.empty(3),
    'RD':np.empty(3),
    'WC':np.empty(3),
    'RPA1':np.empty(3),
    'RPA2':np.empty(3),
    'CP':np.empty(3),
    'WSP':np.empty(3)
}

for k,v in dict_FL.items():
    dict_FR[k]=(np.array([[1,0,0],[0,-1,0],[0,0,1]])@v)

#FRONT LEFT CONSTANTS
FL_k_U=(dict_FL['UF']-dict_FL['UA'])/(np.linalg.norm(dict_FL['UF']-dict_FL['UA']))
FL_k_L=(dict_FL['LF']-dict_FL['LA'])/(np.linalg.norm(dict_FL['LF']-dict_FL['LA']))
FL_d1=np.linalg.norm(dict_FL['UBJ']-dict_FL['LBJ'])
FL_d2=np.linalg.norm(dict_FL['TRO']-dict_FL['TRI'])
FL_d3=np.linalg.norm(dict_FL['PRO']-dict_FL['PRI'])

#FRONT RIGHT CONSTANTS
FR_k_U=(dict_FR['UF']-dict_FR['UA'])/(np.linalg.norm(dict_FR['UF']-dict_FR['UA']))
FR_k_L=(dict_FR['LF']-dict_FR['LA'])/(np.linalg.norm(dict_FR['LF']-dict_FR['LA']))
FR_d1=np.linalg.norm(dict_FR['UBJ']-dict_FR['LBJ'])
FR_d2=np.linalg.norm(dict_FR['TRO']-dict_FR['TRI'])
FR_d3=np.linalg.norm(dict_FR['PRO']-dict_FR['PRI'])

#REAR SUSPENSION 
rearsus_df=pd.read_csv(r"C:\PROGRAMMING\Codes\SUSPENSION KINEMATICS SIMULATOR\CODE REPOSITORY\data\REARSUSP.csv")
rearsus_df.set_index('LABEL',inplace=True)

dict_RL=dict()
dict_RL={
    'LF':np.empty(3),
    'LA':np.empty(3),
    'LBJ':np.empty(3),
    'UF':np.empty(3),
    'UA':np.empty(3),
    'UBJ':np.empty(3),
    'PRO':np.empty(3),
    'PRI':np.empty(3),
    'TRO':np.empty(3),
    'TRI':np.empty(3),
    'DC':np.empty(3),
    'RD':np.empty(3),
    'WC':np.empty(3),
    'RPA1':np.empty(3),
    'RPA2':np.empty(3),
    'CP':np.empty(3),
    'WSP':np.empty(3)
}

for k,v in dict_RL.items():
    temp=rearsus_df.loc[k,['X','Y','Z']].values
    temp=(np.array([[-1,0,0],[0,1,0],[0,0,1]])@(temp))
    temp[0]=temp[0]+1000
    dict_RL[k]=temp

dict_RR=dict()
dict_RR={
    'LF':np.empty(3),
    'LA':np.empty(3),
    'LBJ':np.empty(3),
    'UF':np.empty(3),
    'UA':np.empty(3),
    'UBJ':np.empty(3),
    'PRO':np.empty(3),
    'PRI':np.empty(3),
    'TRO':np.empty(3),
    'TRI':np.empty(3),
    'DC':np.empty(3),
    'RD':np.empty(3),
    'WC':np.empty(3),
    'RPA1':np.empty(3),
    'RPA2':np.empty(3),
    'CP':np.empty(3),
    'WSP':np.empty(3)
}

for k,v in dict_RL.items():
    dict_RR[k]=(np.array([[1,0,0],[0,-1,0],[0,0,1]])@v)

#REAR LEFT CONSTANTS
RL_k_U=(dict_RL['UF']-dict_RL['UA'])/(np.linalg.norm(dict_RL['UF']-dict_RL['UA']))
RL_k_L=(dict_RL['LF']-dict_RL['LA'])/(np.linalg.norm(dict_RL['LF']-dict_RL['LA']))
RL_d1=np.linalg.norm(dict_RL['UBJ']-dict_RL['LBJ'])
RL_d2=np.linalg.norm(dict_RL['TRO']-dict_RL['TRI'])
RL_d3=np.linalg.norm(dict_RL['PRO']-dict_RL['PRI'])

#REAR RIGHT CONSTANTS
RR_k_U=(dict_RR['UF']-dict_RR['UA'])/(np.linalg.norm(dict_RR['UF']-dict_RR['UA']))
RR_k_L=(dict_RR['LF']-dict_RR['LA'])/(np.linalg.norm(dict_RR['LF']-dict_RR['LA']))
RR_d1=np.linalg.norm(dict_RR['UBJ']-dict_RR['LBJ'])
RR_d2=np.linalg.norm(dict_RR['TRO']-dict_RR['TRI'])
RR_d3=np.linalg.norm(dict_RR['PRO']-dict_RR['PRI'])

#static values of our car:
front_static_camber=-1.6
front_static_toe=0.0
rear_static_camber=-1.4
rear_static_toe=0.0

#other gen values
wheelbase=1650
cog_height=285
front_brake_bias=0.6