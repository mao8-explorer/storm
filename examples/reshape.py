import numpy  as np


a = np.ones([3,2,4,3])

for i in range(3):
    for j in range(2):
        for k in range(4):
            for w in range(3):
                a[i,j,k,w] = (k+1)*(w+1) + i + j





print(a)

a.reshape(-1,4)
