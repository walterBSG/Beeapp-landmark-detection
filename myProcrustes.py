import re
import numpy as np
from sklearn.model_selection import train_test_split

from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

from numpy import linalg as LA


def tpsReader(fileName, label):
    with open(fileName, 'r') as myfile:
       data=myfile.read().replace('\n', ' ')

    rLI = re.findall(r"ID=\d+-\d-\d", data)

    data = data.split(' ')
    data = [ x for x in data if 'ID' not in x ]
    data = [ x for x in data if 'LM' not in x ]  
    data = [ x for x in data if 'IM' not in x ]

    data = ' '.join(data)

    r = re.findall(r"[-+]?\d*\.\d+", data)

    i = np.reshape(r, (-1, 19*2))

    o = np.zeros((i.shape[0], 1)) + label
    #Xy = np.hstack((i, o)).astype(float)
    #np.savetxt('LI.tab', dataLI, fmt='%f', delimiter='\t')
    return i, o



def pRef(X):
     ref = X[0].copy()
     ref = ref.astype(float)
     #calculating the procrustes reference
     while(True):
        ref = np.reshape(ref, (-1, 2))
        mtx2All = np.empty([0, 2])
        for i in range(0, X.shape[0]):   
           x = np.reshape(X[i], (-1, 2))
           #mtx1, mtx2 ,_ = procrustes(ref, x)
       
           #xn = x - np.mean(x, 0)
           #mtx2 = xn / np.linalg.norm(xn)
           mtx1, mtx2 ,_ = procrustes(ref, x)
           mtx2All = np.vstack((mtx2All, mtx2))  # it is slow!!

        mtx2All = np.reshape(mtx2All, (-1, 38))
        #print('mtx2All = ', mtx2All.shape)
        avg = np.mean(mtx2All, axis=0)
        ref = np.reshape(ref, (-1, 38))
    
        d = LA.norm(np.subtract(ref, avg))
        if d>0.02:
          ref = avg
          print('loop for procrustes mean d=', d)
          continue
        else:
          ref = avg
          break

     return ref



def projectTg(ref2, x):
       z = x - np.mean(x, axis=0)
       z /= np.linalg.norm(z)
       R, _ = orthogonal_procrustes(z, ref2)
       I = np.identity(19)
       #c, s = np.cos(theta), np.sin(theta)

       #R = np.array(((c,-s), (s, c)))
       #R = np.array(((c),(s))).T
       
       G = np.matmul(ref2, ref2.T)
       mtx2 = I-G
       mtx2 = np.matmul(mtx2, z)
       mtx2 = np.matmul(R, mtx2.T).T

       return mtx2
 


#1 select a random shape as a mean shape (usually the first shape in the set is taken)
#2 Align all the other shapes to it
#3 Calculate a new mean shape, compare it to the old shape (e.g. by subtracting one from the other)
#4 If the difference in mean shapes is above some margin, align the new mean to the old mean and return to step 2.

def projOnRef(ref, X):
    # projecting the training dataset acording the procrustes reference
    ref2 = np.reshape(ref, (-1, 2))
    mtx2All = np.empty([0, 2])
    for i in range(0, X.shape[0]):   
           x = np.reshape(X[i], (-1, 2))

           #R, sca = orthogonal_procrustes(x, ref2)
           #mtx2 = np.matmul(x, R)
       
           mtx1, mtx2 ,_ = procrustes(ref2, x)
       
       
           #theta = np.arccos(np.dot(mtx1, mtx2.T))
           #print('theta shape=', theta.shape)
       
           #mtx2 = project(ref2, x)
           mtx2All = np.vstack((mtx2All, mtx2))

    mtx2All = np.reshape(mtx2All, (-1, 38))
    #yt = np.reshape(y_train, (-1, 1))
    #dataTrainNorm = np.hstack((mtx2All, yt)).astype(float)
    return mtx2All