"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper 
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(p1, p2, M):
    # Replace pass by your implementation
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
    
    np.savez('q2_1.npz', F=F, M=M)
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
    func = lambda x: np.linalg.det((x * F1) + ((1 - x) * F2))
    x0 = func(0)
    x1 = (2 * (func(1) - func(-1)) / 3.0) - ((func(2) - func(-2)) / 12.0)
    x2 = (0.5 * func(1)) + (0.5 * func(-1)) - func(0)
    x3 = func(1) - x0 - x1 - x2
    alphas = np.roots([x3,x2,x1,x0])
    alphas = np.real(alphas[np.isreal(alphas)])
    final = []
    for a in alphas:
        F = a * F1 + (1-a) * F2
        u, s, vh = np.linalg.svd(np.array(F))
        s[2] = 0
        F = u @ np.diag(s) @ vh
        F = helper.refineF(F, pts1, pts2)
        T = np.diag((1 / M, 1 / M, 1))
        F = T.T @ F @ T
        final.append(F)
    final = np.array(final)
    return final[2]

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
    # Replace pass by your implementation
    pass

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    pass

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

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
    # Replace pass by your implementation
    pass

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
    # Replace pass by your implementation
    pass
