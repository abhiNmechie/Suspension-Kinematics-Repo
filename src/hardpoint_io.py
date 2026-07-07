#harpoint naming convention
#RULE: "CORNER_SYMBOL_AXIS"
#CORNERS: {FL, FR, RL, RR}
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
    temp=(np.array([[-1,0,0],[0,1,0],[0,0,1]]@temp))
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

k_U_Left=(dict_FL['UA']-dict_FL['UF'])/(np.linalg.norm(dict_FL['UA']-dict_FL['UF']))
k_L_Left=k_L=(dict_FL['LA']-dict_FL['LF'])/(np.linalg.norm(dict_FL['LA']-dict_FL['LF']))
d1=np.linalg.norm(dict_FL['UBJ']-dict_FL['LBJ'])
d2=np.linalg.norm(dict_FL['TRO']-dict_FL['TRI'])
d3=np.linalg.norm(dict_FL['PRO']-dict_FL['PRI'])

print(dict_FL['LBJ'])