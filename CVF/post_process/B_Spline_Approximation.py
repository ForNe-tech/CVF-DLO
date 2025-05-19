import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import json
import os
import random

class BS_curve(object):

    def __init__(self, n, p, cp=None, knots=None):
        self.n = n  # n+1 control points >>> p0,p1,,,pn
        self.p = p
        if cp:
            self.cp = cp
            self.u = knots
            self.m = knots.shape[0] - 1  # m+1 knots >>> u0,u1,,,nm
        else:
            self.cp = None
            self.u = None
            self.m = None

        self.paras = None

    def check(self):
        if self.m == self.n + self.p + 1:
            return 1
        else:
            return 0

    def coeffs(self, uq):
        # n+1 control points >>> p0,p1,,,pn
        # m+1 knots >>> u0,u1,,,nm
        # algorithm is from https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-curve-coef.html

        # N[] holds all intermediate and the final results
        # in fact N is longer than control points,this is just to hold the intermediate value
        # at last, we juest extract a part of N,that is N[0:n+1]
        N = np.zeros(self.m + 1, dtype=np.float64)

        # rule out special cases Important Properties of clamped B-spline curve
        if uq == self.u[0]:
            N[0] = 1.0
            return N[0:self.n + 1]
        elif uq == self.u[self.m]:
            N[self.n] = 1.0
            return N[0:self.n + 1]

        # now u is between u0 and um
        # first find k uq in span [uk,uk+1)
        check = uq - self.u
        ind = check >= 0
        k = np.max(np.nonzero(ind))
        # sk >>> multiplicity of u[k]
        sk = np.sum(self.u == self.u[k])

        N[k] = 1.0  # degree 0
        # degree d goes from 1 to p
        for d in range(1, self.p + 1):
            r_max = self.m - d - 1  # the maximum subscript value of N in degree d,the minimum is 0
            if k - d >= 0:
                if self.u[k + 1] - self.u[k - d + 1]:
                    N[k - d] = (self.u[k + 1] - uq) / (self.u[k + 1] - self.u[k - d + 1]) * N[
                        k - d + 1]  # right (south-west corner) term only
                else:
                    N[k - d] = (self.u[k + 1] - uq) / 1 * N[k - d + 1]  # right (south-west corner) term only

            for i in range(k - d + 1, (k - 1) + 1):
                if i >= 0 and i <= r_max:
                    Denominator1 = self.u[i + d] - self.u[i]
                    Denominator2 = self.u[i + d + 1] - self.u[i + 1]
                    # 0/0=0
                    if Denominator1 == 0:
                        Denominator1 = 1
                    if Denominator2 == 0:
                        Denominator2 = 1

                    N[i] = (uq - self.u[i]) / (Denominator1) * N[i] + (self.u[i + d + 1] - uq) / (Denominator2) * N[
                        i + 1]

            if k <= r_max:
                if self.u[k + d] - self.u[k]:
                    N[k] = (uq - self.u[k]) / (self.u[k + d] - self.u[k]) * N[k]
                else:
                    N[k] = (uq - self.u[k]) / 1 * N[k]

        return N[0:self.n + 1]

    def De_Boor(self, uq):
        # Input: a value u
        # Output: the point on the curve, C(u)

        # first find k uq in span [uk,uk+1)
        check = uq - self.u
        ind = check >= 0
        k = np.max(np.nonzero(ind))

        # inserting uq h times
        if uq in self.u:
            # sk >>> multiplicity of u[k]
            sk = np.sum(self.u == self.u[k])
            h = self.p - sk
        else:
            sk = 0
            h = self.p

        # rule out special cases
        if h == -1:
            if k == self.p:
                return np.array(self.cp[0])
            elif k == self.m:
                return np.array(self.cp[-1])

        # initial values of P(affected control points) >>> Pk-s,0 Pk-s-1,0 ... Pk-p+1,0
        P = self.cp[k - self.p:k - sk + 1]
        P = P.copy()
        dis = k - self.p  # the index distance between storage loaction and varibale i
        # 1-h

        for r in range(1, h + 1):
            # k-p >> k-sk
            temp = []  # uesd for Storing variables of the current stage
            for i in range(k - self.p + r, k - sk + 1):
                a_ir = (uq - self.u[i]) / (self.u[i + self.p - r + 1] - self.u[i])
                temp.append((1 - a_ir) * P[i - dis - 1] + a_ir * P[i - dis])
            P[k - self.p + r - dis:k - sk + 1 - dis] = np.array(temp)
        # the last value is what we want
        return P[-1]

    def bs(self, us):
        y = []
        for x in us:
            y.append(self.De_Boor(x))
        y = np.array(y)
        return y

    def estimate_parameters(self, data_points, method="centripetal"):
        pts = data_points.copy()
        N = pts.shape[0]
        w = pts.shape[1]
        Li = []
        for i in range(1, N):
            Li.append(np.sum([pts[i, j] ** 2 for j in range(w)]) ** 0.5)
        L = np.sum(Li)

        t = [0]
        for i in range(len(Li)):
            Lki = 0
            for j in range(i + 1):
                Lki += Li[j]
            t.append(Lki / L)
        t = np.array(t)
        self.paras = t
        ind = t > 1.0
        t[ind] = 1.0
        return t

    def get_knots(self, method="average"):

        knots = np.zeros(self.p + 1).tolist()

        paras_temp = self.paras.copy()
        # m = n+p+1
        self.m = self.n + self.p + 1
        # we only need m+1 knots
        # so we just select m+1-(p+1)-(p+1)+(p-1)+1+1  paras to average
        num = self.m - self.p  # select n+1 paras

        ind = np.linspace(0, paras_temp.shape[0] - 1, num)
        ind = ind.astype(int)
        paras_knots = paras_temp[ind]

        for j in range(1, self.n - self.p + 1):
            k_temp = 0
            # the maximun of variable i is n-1
            for i in range(j, j + self.p - 1 + 1):
                k_temp += paras_knots[i]
            k_temp /= self.p
            knots.append(k_temp)

        add = np.ones(self.p + 1).tolist()
        knots = knots + add
        knots = np.array(knots)
        self.u = knots
        self.m = knots.shape[0] - 1
        return knots

    def set_paras(self, parameters):
        self.paras = parameters

    def set_knots(self, knots):
        self.u = knots

    def approximation(self, pts):
        ## Obtain a set of parameters t0, ..., tn
        # pts_paras = self.estimate_parameters(pts)
        ## knot vector U;
        # knots = self.get_knots()
        num = pts.shape[0] - 1  # (num+1) is the number of EWD points

        P = np.zeros((self.n + 1, pts.shape[1]), dtype=np.float64)  # n+1 control points
        P[0] = pts[0]
        P[-1] = pts[-1]

        # compute N
        N = []
        for uq in self.paras:
            N_temp = self.coeffs(uq)
            N.append(N_temp)
        N = np.array(N)

        Q = [0]  # hold the location
        for k in range(1, num - 1 + 1):
            Q_temp = pts[k] - N[k, 0] * pts[0] - N[k, self.n] * pts[-1]
            Q.append(Q_temp)

        b = [0]
        for i in range(1, self.n - 1 + 1):
            b_temp = 0
            for k in range(1, num - 1 + 1):
                b_temp += N[k, i] * Q[k]
            b.append(b_temp)

        b = b[1::]
        b = np.array(b)

        N = N[:, 1:(self.n - 1) + 1]
        A = np.dot(N.T, N)
        cpm = np.linalg.solve(A, b)
        P[1:self.n] = cpm
        self.cp = P
        return P

if __name__ == "__main__":

    # 读取线缆路径散点
    pathdir = '../../EWD/LAB_imgs_1028_DLO/path2D/'
    bsdir = '../../EWD/LAB_imgs_1028_DLO/path2D_bs/'
    visdir = '../../EWD/LAB_imgs_1028_DLO/path2D_vis/'

    os.makedirs(bsdir, exist_ok=True)
    os.makedirs(visdir, exist_ok=True)

    scene_list = [''] + [str(i) for i in range(29)]

    for scene_label in scene_list:

        jsonpath = pathdir + 'temp{}_paths2D.json'.format(scene_label)

        if not os.path.exists(jsonpath):
            continue

        with open(jsonpath, 'r', encoding='utf-8') as file:
            path_dict = json.load(file)

        bs_paras = {}
        bs_paras_path = bsdir + 'temp{}_bs2D.json'.format(scene_label)
        bs_path2D = {}
        bs_path2D_path = bsdir + 'temp{}_paths2D_bs.json'.format(scene_label)

        for label, path in zip(path_dict.keys(), path_dict.values()):

            num = len(path)

            cp_num = max(round(num / 100), 6)
            vp_num = max(round(num / 3), 20)

            path_arr = np.array(path)

            yy = path_arr[:, 0]
            # yy = -yy + 504    # C#读图顺序与python不同
            xx = path_arr[:, 1]

            bs = BS_curve(cp_num, 3)

            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111)
            # ax.scatter(xx, yy, c='green')

            data = np.array([xx, yy]).T
            paras = bs.estimate_parameters(data)
            knots = bs.get_knots()

            if bs.check():
                cp = bs.approximation(data)

            uq = np.linspace(0, 1, vp_num)
            y = bs.bs(uq)
            ax.plot(y[:, 0], y[:, 1], '-r')
            # ax.plot(cp[:, 0], cp[:, 1], '-b*')
            plt.xlim(0, 896)
            plt.ylim(0, 504)
            plt.axis('off')
            # plt.show()
            plt.savefig(visdir + 'temp{}_{}_path2D_bs.png'.format(scene_label, label))

            bs_paras[label] = {'knots': knots.tolist(), 'cp': cp.tolist()}
            y[:, [0, 1]] = y[:, [1, 0]]
            bs_path2D[label] = y.tolist()

        with open(bs_paras_path, 'w', encoding='utf-8') as f1:
            json.dump(bs_paras, f1, ensure_ascii=False, indent=4)
        with open(bs_path2D_path, 'w', encoding='utf-8') as f2:
            json.dump(bs_path2D, f2, ensure_ascii=False, indent=4)