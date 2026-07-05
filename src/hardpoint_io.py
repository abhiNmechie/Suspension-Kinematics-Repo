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
def mods(arr):
    arr2=np.empty(3)
    arr2[[0,1]]=(-arr[[0,1]])
    arr2[0]=arr2[0]+1000
    return arr2

frontsus_df=pd.read_csv(r"C:\PROGRAMMING\Codes\SUSPENSION KINEMATICS SIMULATOR\CODE REPOSITORY\data\frontsusFINAL2.txt")
frontsus_df.set_index('LABEL',inplace=True)

mapping=dict()
UA,UF,UBJ,LA,LF,LBJ,TRI,TRO,WC,CP,PRO,RPA1,RPA2,RPR,RD,DC=[np.empty(3) for _ in range(16)]
mapping={
    'Lower wishbone front pivot':LF,
    'Lower wishbone rear pivot':LA,
    'Lower wishbone outer ball joint':LBJ,
    'Upper wishbone front pivot':UF,
    'Upper wishbone rear pivot':UA,
    'Upper wishbone outer ball joint':UBJ,
    'Push rod wishbone end':PRO,
    'Push rod rocker end':RPR,
    'Outer track rod ball joint':TRO,
    'Inner track rod ball joint':TRI,
    'Damper to body point':DC,
    'Damper to rocker point':RD,
    'Wheel centre point':WC,
    'Rocker axis 1st point':RPA1,
    'Rocker axis 2nd point':RPA2,
    'Contact Patch':CP
}
i=0
for k,v in mapping.items():
    temp=frontsus_df.loc[k,['X','Y','Z']].values
    v=mods(temp)
    i=i+1
    print(v)
print(i)