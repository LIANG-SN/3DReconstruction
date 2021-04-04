"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper 
import sys
import math
from scipy.ndimage import gaussian_filter
import scipy
import cv2
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(p1, p2, M): #  inverse p1 & p2 ??

    assert(p1.shape[0]==p2.shape[0])
    assert(p1.shape[1]==2)

    #review this
    p1 = p1 / M
    p2 = p2 / M
    
    # todo: relation between p1 and p2
    n = p1.shape[0]
    A = np.zeros((n, 9))
    A[0:n, 0] = p2[:, 0] * p1[:, 0]
    A[0:n, 1] = p2[:, 0] * p1[:, 1]
    A[0:n, 2] = p2[:, 0]
    A[0:n, 3] = p2[:, 1] * p1[:, 0]
    A[0:n, 4] = p2[:, 1] * p1[:, 1]
    A[0:n, 5] = p2[:, 1]
    A[0:n, 6] = p1[:, 0]
    A[0:n, 7] = p1[:, 1]
    A[0:n, 8] = 1

    u, s, vh = np.linalg.svd(np.array(A))
    F = vh[-1].reshape(3,3)

    u, s, vh = np.linalg.svd(np.array(F))
    s[2] = 0
    F = u @ np.diag(s) @ vh
    F = helper.refineF(F, p1, p2)
    T = np.diag((1 / M, 1 / M, 1))
    F = T.T @ F @ T
    
    # np.savez('q2_1.npz', F=F, M=M)  # resave this
    return F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):

    pts1 = pts1 / M
    pts2 = pts2 / M
    A = [pts2[:,0]*pts1[:,0], pts2[:,0]*pts1[:,1], pts2[:,0], pts2[:,1]*pts1[:,0], \
        pts2[:,1]*pts1[:,1], pts2[:,1], pts1[:,0], pts1[:,1], np.ones_like(pts1[:,0])]
    A = np.asarray(A).T
    u, s, v = np.linalg.svd(A)
    F1 = v[-1].reshape(3,3)
    F2 = v[-2].reshape(3,3)
    # use the property of determinant
    det = lambda x: np.linalg.det((x * F1) + ((1 - x) * F2))
    a0 = det(0)
    a1 = (2 * (det(1) - det(-1)) / 3.0) - ((det(2) - det(-2)) / 12.0)
    a2 = (0.5 * det(1)) + (0.5 * det(-1)) - det(0)
    a3 = det(1) - a0 - a1 - a2
    alphas = np.roots([a3,a2,a1,a0])
    alphas = np.real(alphas[np.isreal(alphas)])
    Fs = []
    for a in alphas:
        F = a * F1 + (1-a) * F2
        u, s, vh = np.linalg.svd(np.array(F))
        s[2] = 0
        F = u @ np.diag(s) @ vh
        F = helper.refineF(F, pts1, pts2)
        T = np.diag((1 / M, 1 / M, 1))
        F = T.T @ F @ T
        Fs.append(F)
    Fs = np.array(Fs)
    # uncomment this to save F
    # np.savez('q2_2.npz', F=final, M=M, pts1=pts1, pts2=pts2)
    return Fs


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return K2.T @ F @ K1

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    P = np.zeros((pts1.shape[0], 4))
    p1 = np.zeros((pts1.shape[0], 3))
    p2 = np.zeros((pts1.shape[0], 3))
    err = 0
    for i in range(pts1.shape[0]):
        A = np.zeros((4, 4))
        A[0] = pts1[i, 1] * C1[2] - C1[1]
        A[1] = C1[0] - pts1[i, 0] * C1[2]
        A[2] = pts2[i, 1] * C2[2] - C2[1]
        A[3] = C2[0] - pts2[i, 0] * C2[2]
        u, s, v = np.linalg.svd(A)
        P[i] = v[-1]
        P[i] /= P[i, 3]
        p1[i] = C1 @ P[i]
        p2[i] = C2 @ P[i]
        p1[i] /= p1[i, 2]
        p2[i] /= p2[i, 2]
        err += np.sum(np.square(pts1[i] - p1[i, 0:2])) + np.sum(np.square(pts2[i] - p2[i, 0:2]))
    return P[:, :3], err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    window = 3
    p1 = np.ones(3)
    p1[:2] = [x1, y1]
    l = F @ p1
    
    w1 = im1.shape[1]
    h1 = im1.shape[0]
    w2 = im2.shape[1]
    h2 = im2.shape[0]
    im1Patch = np.zeros(((h1 + (window // 2) * 2), (w1 + (window // 2) * 2)))
    im1Patch[window // 2 : -window // 2 + 1, window // 2 : -window // 2 + 1] = im1
    im2Patch = np.zeros(((h2 + (window // 2) * 2), (w2 + (window // 2) * 2)))
    im2Patch[window // 2 : -window // 2 + 1, window // 2 : -window // 2 + 1] = im2
    error = 20010602
    x2 = 0
    y2 = 0
    # use x = -b/a * y - c/a
    k = - l[1] / l[0]
    b = - l[2] / l[0]
    for i in range(h2):
        y2Temp = i
        x2Temp = (int)(k * i + b)
        if x2Temp >= 0 and x2Temp < w2 and math.sqrt((x1-x2Temp)**2 + (y1-y2Temp)**2) <= 35:
            patch1 = im1Patch[y1 - window // 2 + window // 2 : y1 + window // 2 + window // 2 + 1, \
                x1 - window // 2 + window // 2 : x1 + window // 2 + window // 2 + 1]
            patch2 = im2Patch[y2Temp - window // 2 + window // 2 : y2Temp + window // 2 + window // 2 + 1,\
                 x2Temp - window // 2 + window // 2 : x2Temp + window // 2 + window // 2 + 1]
            patch = gaussian_filter(patch1 - patch2, 1)
            err = np.linalg.norm(patch)
            if err < error:
                error = err
                x2 = x2Temp
                y2 = y2Temp
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return x2, y2


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):

    numIteration = 300
    tol = 2
    maxInliers = -1
    bestF = np.zeros((3,3))
    bestInliers = []
    # ransac loop
    for i in range(numIteration):
        # random choose 7 points
        sample = np.random.randint(0, pts1.shape[0], 7)
        # sample points
        p1 = pts1[sample]
        p2 = pts2[sample]
        F = sevenpoint(p1, p2, M)
        if F.ndim == 3:
            F = F[-1]
        numInliers = 0
        inliers = []
        # check inliers, keep H if max
        for k in range(pts1.shape[0]):
            v = np.array([pts1[k, 0], pts1[k, 1], 1])
            l = F @ v
            s = np.sqrt(l[0]**2+l[1]**2)
            if s==0:
                print('Zero line vector in displayEpipolar')
            l = l/s
            p2 = np.array([pts2[k, 0], pts2[k, 1], 1])
            dist = abs(l @ p2.T)
            if k in sample:
                continue
            elif dist < tol:
                numInliers += 1
                inliers.append(k)
        if numInliers > maxInliers:
            maxInliers = numInliers
            bestF = F
            bestInliers = np.append(sample, inliers)
    bestInliers = np.array(bestInliers, dtype=int)
    # recompute best F with inliers
    bestF = eightpoint(pts1[bestInliers], pts2[bestInliers], M)
    inliersFlag = np.zeros(pts1.shape[0])
    inliersFlag[bestInliers] = 1
    return bestF, inliersFlag

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(rodrigues_vec):
    if rodrigues_vec.ndim == 2:
        rodrigues_vec = rodrigues_vec.reshape((rodrigues_vec.shape[0],))
    # print(rodrigues_vec.shape)
    theta = np.linalg.norm(rodrigues_vec)
    if theta < sys.float_info.epsilon:              
        rotation_mat = np.eye(3, dtype=float)
    else:
        r = rodrigues_vec / theta
        I = np.eye(3, dtype=float)
        r_rT = np.array([
            [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
        ])
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ])
        rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    return rotation_mat 

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    return cv2.Rodrigues(R)[0]

'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    residuals = np.zeros((p1.shape[0] * 4,))
    C1 = K1 @ M1
    M2 = np.zeros((3, 4))
    M2[:, :3] = rodrigues(x[0 : 3])
    M2[:, 3] = x[3:6]
    C2 = K2 @ M2

    for i in range(p1.shape[0]):   
        P = np.append(x[6 + i * 3 : 6 + i * 3 + 3], 1)
        p1_ = C1 @ P
        p2_ = C2 @ P
        p1_ = (p1_ / p1_[-1])[0:2]
        p2_ = (p2_ / p2_[-1])[0:2]
        residuals[4 * i + 0] = (p1_[0] - p1[i, 0]) 
        residuals[4 * i + 1] = (p1_[1] - p1[i, 1])
        residuals[4 * i + 2] = (p2_[0] - p2[i, 0])
        residuals[4 * i + 3] = (p2_[1] - p2[i, 1])
    residuals = residuals.reshape((residuals.shape[0], 1))
    return residuals

def rodriguesResidualCall(x, K1, M1, p1, K2, p2):
    residuals = rodriguesResidual(K1, M1, p1, K2, p2, x)
    residuals = residuals.reshape((residuals.shape[0], ))
    # print(residuals.shape)
    return residuals
'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    
    r = invRodrigues(M2_init[:, 0:3])
    t = M2_init[:,-1]
    x = np.concatenate((r.flatten(), t.flatten(), P_init.flatten()))

    errRaw = rodriguesResidual(K1, M1, p1, K2, p2, x)
    errRaw = sum(np.square(errRaw))
    print('raw err: ', errRaw)

    op = scipy.optimize.least_squares(rodriguesResidualCall, x, args=(K1, M1, p1, K2, p2))
    finalx = op.x
    # print(finalx.shape)

    M2 = np.zeros((3, 4))
    M2[:, :3] = rodrigues(finalx[0 : 3])
    M2[:, 3] = finalx[3:6]

    P2 = finalx[6:].reshape(p1.shape[0], 3)
    errFinal = rodriguesResidual(K1, M1, p1, K2, p2, finalx)
    errFinal = sum(np.square(errFinal))
    print('final err: ', errFinal)
    return M2, P2
