'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import submission as sub
import numpy as np
import cv2
import matplotlib.pyplot as plt

# load image

im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png')

# load camera parameters

K = np.load('../data/intrinsics.npz')
K1 = K['K1']
F = np.load('q2_1.npz')['F']
M1 = np.zeros((3,4))
M1[0, 0] = 1
M1[1, 1] = 1
M1[2, 2] = 1
C1 = K1 @ M1
M2 = np.load('q3_3.npz')['M2']
C2 = np.load('q3_3.npz')['C2']

# load points data

data = np.load('../data/templeCoords.npz')
x1 = data['x1']
y1 = data['y1']
pts1 = np.concatenate((x1, y1), axis=1)
pts2 = []
for p in pts1:
    x2, y2 = sub.epipolarCorrespondence(im1, im2, F, p[0], p[1])
    pts2.append([x2, y2])
pts2 = np.array(pts2)

# trangulate

P, err = sub.triangulate(C1, pts1, C2, pts2)

# save data

# np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

# plot

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], marker='+')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()