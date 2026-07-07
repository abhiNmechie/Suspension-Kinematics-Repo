import numpy as np

a1=np.array([1,2,3])
a2=np.array([2,3,4])

res=np.dot(a1,a2)
a3=np.array([3,4,5])
res2=np.dot(res,a3)
print(res2)