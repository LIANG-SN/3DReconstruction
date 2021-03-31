import numpy as np
import submission as src
import helper
import cv2
if __name__ == '__main__':
    # test q2.1
    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    pts = np.concatenate((pts1, pts2))
    M = np.max(pts)
    data = np.load('q2_1.npz')
    F = data['F']
    # uncomment this to save data
    # F = src.eightpoint(pts1, pts2, M)
    # randomly select 7 points
    # sample = np.random.randint(0, pts1.shape[0]-1, 7)
    # F = src.sevenpoint(pts1[sample, :], pts2[sample, :], M)
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')
    helper.displayEpipolarF(im1, im2, F)

    # 3.2

    
    
    # M1 = np.zeros((3, 4))
    # M1[0, 0] = 1
    # M1[1, 1] = 0
    # M1[2, 2] = 0
    # C1 = K1 @ M1
    # C2 = K2 @ M2
    # P, err = src.triangulate(C1, pts1, C2, pts2)
    # print(err)