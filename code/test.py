import numpy as np
import submission as src
import helper
import cv2
import matplotlib.pyplot as plt
from findM2 import findM2
if __name__ == '__main__':
    # test q2.1
    
    # data = np.load('../data/some_corresp.npz')
    # pts1 = data['pts1']
    # pts2 = data['pts2']
    # pts = np.concatenate((pts1, pts2))
    # M = np.max(pts)
    
    # # load F directly
    # data = np.load('q2_1.npz')
    # F = data['F']
    # print(F)
    
    # # uncomment this to save data
    # # F = src.eightpoint(pts1, pts2, M)

    # im1 = cv2.imread('../data/im1.png')
    # im2 = cv2.imread('../data/im2.png')
    # helper.displayEpipolarF(im1, im2, F)


    # test q2.2

    data = np.load('../data/some_corresp.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    pts = np.concatenate((pts1, pts2))
    M = np.max(pts)

    # load F directly
    # data = np.load('q2_2.npz')
    # Fs = data['F']
    # print(Fs)
    
    # uncomment this to save data
    # randomly select 7 points
    # sample = np.random.randint(0, pts1.shape[0]-1, 7)
    # Fs = src.sevenpoint(pts1[sample, :], pts2[sample, :], M)
    
    # print(Fs.shape)
    # im1 = cv2.imread('../data/im1.png')
    # im2 = cv2.imread('../data/im2.png')
    # if Fs.ndim == 3:
    #     for F in Fs:
    #         helper.displayEpipolarF(im1, im2, F)
    # else:
    #     helper.displayEpipolarF(im1, im2, Fs)

    # 3.2
    
    # M1 = np.zeros((3, 4))
    # M1[0, 0] = 1
    # M1[1, 1] = 0
    # M1[2, 2] = 0
    # C1 = K1 @ M1
    # C2 = K2 @ M2
    # P, err = src.triangulate(C1, pts1, C2, pts2)
    # print(err)


    # 4.1
    # data = np.load('q2_1.npz')
    # F = data['F']
    # pts1, pts2 = helper.epipolarMatchGUI(im1, im2, F)
    # save pts
    # assert pts1.shape[1] == 2
    # assert pts2.shape[1] == 2
    # np.savez('q4_1.npz', F=F, pts1=pts1, pts2=pts2)
    # data = np.load('q4_1.npz')
    # print(data.files)

    # test 5.1
    # data = np.load('../data/some_corresp_noisy.npz')
    # pts1 = data['pts1']
    # pts2 = data['pts2']
    # pts = np.concatenate((pts1, pts2))
    # M = np.max(pts)
    # F, inliers = src.ransacF(pts1, pts2, M)
    # im1 = cv2.imread('../data/im1.png')
    # im2 = cv2.imread('../data/im2.png')
    # helper.displayEpipolarF(im1, im2, F)

    # test 5.2
    # R = np.ones((3,3))
    # r = src.invRodrigues(R)


    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(P[:, 0], P[:, 1], P[:, 2], marker='+')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

    # test 5.3
    # data_noisy = np.load('../data/some_corresp_noisy.npz')
    # pts1 = data_noisy['pts1']
    # pts2 = data_noisy['pts2']
    # pts = np.concatenate((pts1, pts2))
    # M = np.max(pts)
    # F, inliers = src.ransacF(pts1, pts2, M)
    # pts1 = pts1[np.nonzero(inliers)]
    # pts2 = pts2[np.nonzero(inliers)]
    # M2 = findM2(pts1, pts2, F)
    # K = np.load('../data/intrinsics.npz')
    # K1 = K['K1']
    # K2 = K['K2']
    # M1 = np.zeros((3,4))
    # M1[0, 0] = 1
    # M1[1, 1] = 1
    # M1[2, 2] = 1
    # P, err = src.triangulate(K1 @ M1, pts1, K2 @ M2, pts2)
    # M2, P2 = src.bundleAdjustment(K1, M1, pts1, K2, M2, pts2, P)

    # # plot
    # fig = plt.figure()
    # ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    # ax0.scatter(P[:, 0], P[:, 1], P[:, 2], marker='+')
    # ax0.set_xlabel('X')
    # ax0.set_ylabel('Y')
    # ax0.set_zlabel('Z')
    # ax1.scatter(P2[:, 0], P2[:, 1], P2[:, 2], marker='o')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.set_zlabel('Z')
    # plt.show()