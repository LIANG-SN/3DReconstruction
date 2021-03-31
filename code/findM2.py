'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import helper
import submission as sub
import numpy as np


data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']

K = np.load('../data/intrinsics.npz')
K1 = K['K1']
K2 = K['K2']
F = np.load('q2_1.npz')['F']
E = sub.essentialMatrix(F, K1, K2)
M2s = helper.camera2(E) # four possible
R = np.zeros((4, 3, 3))

C2 = np.zeros((4, 3, 4))
C1 = np.zeros((3,4))
C1[0, 0] = 1
C1[1, 1] = 1
C1[2, 2] = 1
C1 = K1 @ C1
for i in range(4):
    C2[i] = K2 @ M2s[:, :, i]
    P, err = sub.triangulate(C1, pts1, C2[i], pts2)
    if (P[:, -1] >= 0).all():
        M2 = M2s[:, :, i]
        print('find', i, err)
        # np.savez('q3_3.npz', M2=M2, C2=C2, P=P)
        # break