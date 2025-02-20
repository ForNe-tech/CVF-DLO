import numpy as np
import cv2
# from skimage.morphology import skeletonize
import skeletonize as sk
import CVF3D.process.utils_CVF as ut
import math

from time import time


class Skeletonize():

    def __init__(self, drop=False, if_debug=False):
        self.drop = drop
        self.kernel_size = 4
        self.merge_size = 20
        self.cmap = self.voc_cmap(N=256, normalized=False)
        self.if_debug = if_debug
        self.total_mean_width = 10

    def voc_cmap(self, N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255 if normalized else cmap
        return cmap

    def run(self, mask, density=7, source_img=None, verbose=False, times=None):

        # =========================================================================
        verbose_print = print if verbose else lambda x: None
        s = times.pop()
        # =========================================================================

        IMG_H, IMG_W = mask.shape[0], mask.shape[1]

        dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        skeleton = sk.skeletonize(mask)

        if self.if_debug:
            cv2.imwrite('data/debug_results/mask.png', mask)
            cv2.imwrite('data/debug_results/skeleton_lee.png', skeleton * 255)

        if source_img is None:
            source_img = np.zeros_like(mask)

        # =========================================================================
        times.append(time()-s)
        verbose_print("Skeletonizing time: {:.5f}".format(times[-1]))
        s = time()
        # =========================================================================

        # 提取端点,交叉点
        # ends = self.extractEnds(skeleton)
        ints = self.extractInts(skeleton)

        if self.if_debug:
            self.showPoints(skeleton, ints=ints)

        ints_dict = self.mergeIntsDict(ints, dist_img)

        # 交叉点置零,将图像分为若干段
        skeleton_f = skeleton.copy()
        skeleton_f[tuple([ints[0], ints[1]])] = 0

        if self.if_debug:
            cv2.imwrite('data/debug_results/skeleton_f.png', skeleton_f)

        # =========================================================================
        times.append(time() - s)
        verbose_print("Keypoint detection time: {:.5f}".format(times[-1]))
        s = time()
        # =========================================================================

        # 提取路线段
        num_labels, labels = cv2.connectedComponents(skeleton_f)

        ints_dict_rf = self.associateLabelsToIntersections(labels, ints_dict)

        skeleton_f, labels, ints_dict_rf = self.constructIntsDict(skeleton_f, labels, dist_img, ints_dict_rf)

        if self.if_debug:
            cv2.imwrite('data/debug_results/skeleton_ints_rf.png', skeleton_f)
            self.showPoints_dict(skeleton_f, ints = ints_dict_rf, merged=False)

        ends = self.extractEndslist(skeleton_f)
        routes, labels, ends = self.extractRoutes(ends, num_labels, labels, skeleton_f, ints_dict_rf)

        if self.if_debug:
           self.showRoutes(routes, skeleton_f, prune=False)

        # =========================================================================
        times.append(time() - s)
        verbose_print("Route detection time: {:.5f}".format(times[-1]))
        s = time()
        # =========================================================================

        # 计算DLO宽度(平均宽度:该段DLO像素宽度的众数;最大宽度:该段DLO像素宽度的最大值),以及整张图像上所有DLO像素的宽度
        routes = self.estimateRoutewidthFromSegment(routes, dist_img, min_px=3)

        # 删去部分毛刺(平均宽度远小于整张图像上所有DLO像素宽度的DLO,DLO长度小于1.3倍该段DLO像素宽度最大值的DLO)
        routes, ends_rf, routes_im, skeleton_rf = self.prune_short_routes(routes, ends, labels, skeleton_f, IMG_W, IMG_H)

        if self.if_debug:
           self.showRoutes(routes, skeleton_f, prune=True)

        # =========================================================================
        times.append(time() - s)
        verbose_print("Split end pruning time: {:.5f}".format(times[-1]))
        s = time()
        # =========================================================================

        # DLO重新转化为图像
        # skeleton_rf, routes_im = self.RoutesToSkeleton(routes, skeleton_f)

        # 重新提取端点
        # ends_rf = self.extractEnds(skeleton_rf)

        if self.if_debug:
            cv2.imwrite('data/debug_results/skeleton_rf.png', skeleton_rf)
            self.showPoints(skeleton_rf, ends_rf)

        # 端点信息绑定(孤立端点,交叉端点,端点隶属DLO编号),端点绑定到交叉点
        ends_dict_rf, ints_dict_rf = self.constructEndsDict_New(routes, routes_im, source_img, ends_rf, ints_dict_rf)

        # 整合交叉点
        ints_dict_m = self.mergeIntsFromRoutes(routes, ints_dict_rf)

        if self.if_debug:
            self.showPoints_dict(skeleton_rf, ints = ints_dict_m)

        # =========================================================================
        times.append(time() - s)
        verbose_print("Information integration time: {:.5f}".format(times[-1]))
        s = time()
        # =========================================================================

        # 计算交叉端点方向
        ends_dict_rf, _ = self.calcRouteDirection(routes_im, ends_dict_rf, 'int')
        # 处理交叉端点
        end_pairs = self.handleIntersections(skeleton_rf, ends_dict_rf, ints_dict_m)

        # # 计算孤立端点方向
        # ends_dict_iso, dir_ends_dict = self.calcRouteDirection(routes_im, ends_dict_rf, 'iso')
        # # 孤立端点匹配
        # end_pairs += self.checkForContinuity(source_img, skeleton_rf, ends_dict_iso, dir_ends_dict)

        # 清除可能出现的重复端点对
        end_pairs = self.cleanEndPairs(end_pairs)

        if self.if_debug:
            # 重新连接
            skeleton_m = self.mergeEnds(skeleton_rf, ends_dict_rf, end_pairs)
            cv2.imwrite('data/debug_results/skeleton_m.png', skeleton_m)

        # =========================================================================
        times.append(time() - s)
        verbose_print("Intersection clustering and matching time: {:.5f}".format(times[-1]))
        times.append(time())
        # =========================================================================

        return skeleton_rf, routes, end_pairs, ends_dict_rf, ints_dict_m, times

    def showPoints_dict(self, skel, ends=None, ints=None, merged=True):
        # back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        if merged:
            back = cv2.cvtColor(skel * 255, cv2.COLOR_GRAY2BGR)
        else:
            back = cv2.cvtColor(skel * 255, cv2.COLOR_GRAY2BGR)
        if ends is not None:
            ends_list = list(zip(ends[1], ends[0]))
            for end in ends_list:
                cv2.circle(back, end, 3, (0, 0, 255))
        if ints is not None:
            for inT in ints.values():
                int_point = inT['point']
                int_radii = round(inT['int_radius']) + self.kernel_size
                cv2.circle(back, (int(int_point[1]), int(int_point[0])), int_radii, (0, 255, 0))
        cv2.imwrite('data/debug_results/show_points_dict_{}.jpg'.format(merged), back)


    def showPoints(self, skel, ends=None, ints=None):
        back = cv2.cvtColor(skel * 255, cv2.COLOR_GRAY2BGR)
        back = ~back
        if ends is not None:
            ends_list = list(zip(ends[1], ends[0]))
            for end in ends_list:
                cv2.circle(back, end, 3, (0, 0, 255))
        if ints is not None:
            ints_list = list(zip(ints[1], ints[0]))
            for int in ints_list:
                cv2.circle(back, int, 3, (0, 255, 0))
        cv2.imwrite('data/debug_results/show_points.jpg', back)

    def showRoutes(self, routes, skel, prune=False, mask=None):
        if mask is not None:
            back_white = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
            back_white = np.ones_like(back) * 255
        for i in routes.keys():
            for point in routes[i]['route']:
                back_white[point[0]][point[1]] = self.cmap[i]
        if prune:
            cv2.imwrite('data/debug_results/show_routes_prune.jpg', back_white)
        else:
            cv2.imwrite('data/debug_results/show_routes.jpg', back_white)

    def showEndsAndRadius(self, skel, ends_dict_rf, routes):
        back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        back = ~back
        for i, end_dict in ends_dict_rf.items():
            end_point = end_dict['point']
            end_route_label = end_dict['route_label'][0]
            radii = int(0.1 * len(routes[end_route_label]))
            cv2.circle(back, (end_point[1], end_point[0]), radii, (0, 255, 0))
        cv2.imwrite('data/debug_results/show_ends_and_radius.jpg', back)

    def mergeInts(self, ints, thre):
        ints_list = list(zip(ints[0], ints[1]))
        ints_rf = []
        for p in ints_list:
            already_in = False
            for v in ints_rf:
                if self.distance2D(p, v) <= thre:
                    already_in = True
            if not already_in:
                ints_rf.append(p)
        return np.array(ints_rf).T

    def mergeIntsDict(self, ints, dist_img):
        ints_list = list(zip(ints[0], ints[1]))
        ints_dict = {}
        i = 0
        for p in ints_list:
            already_in = False
            p_r = dist_img[p[0]][p[1]]
            for idx, int_dict in ints_dict.items():
                v = int_dict['point']
                v_r = int_dict['int_radius']
                dist_pv = self.distance2D(p, v)
                if dist_pv < self.kernel_size:
                    already_in = True
                    nv = ((p[0] + v[0])//2, (p[1] + v[1])//2)
                    int_dict['point'] = nv
                    int_dict['int_radius'] = (p_r + v_r + dist_pv)/2
                    break
            if not already_in:
                ints_dict[i] = {"point": p,
                                "routes_label": [],
                                "int_ends": [],
                                "int_radius": p_r}
                i += 1

        return ints_dict

    def associateLabelsToIntersections(self, labels_im, ints_dict):
        for k, int_dict in ints_dict.items():
            point = int_dict['point']
            radius = int_dict['int_radius']
            window_size = round(radius) + self.kernel_size
            label_cover = labels_im[(point[0] - window_size):(point[0] + window_size),
                                    (point[1] - window_size):(point[1] + window_size)]
            int_dict['routes_label'] = [v for v in np.unique(label_cover) if v != 0]
        return ints_dict

    def RoutesToSkeleton(self, routes, skel):
        back1 = np.zeros_like(skel)
        back2 = np.zeros_like(skel)
        for i in routes.keys():
            for point in routes[i]['route']:
                back1[point[0]][point[1]] = 255
                back2[point[0]][point[1]] = i
        return back1, back2

    def extractRoutes(self, ends, num_labels, labels, skel_img, ints_dict_rf):
        skel = skel_img.astype(bool)
        routes = {}
        ends_rf = []
        for n in range(1, num_labels + 1):
            ends_f = [e for e in ends if labels[tuple([e[1], e[0]])] == n]
            if len(ends_f) == 2:
                curr_pixel = np.array([ends_f[0][1], ends_f[0][0]]).astype('int16')
                route = ut.traverse_skeleton(skel, curr_pixel)
                if len(route) > 0:
                    ends_rf += ends_f
                    rl_conn = []
                    for int_dict in ints_dict_rf.values():
                        if n in int_dict['routes_label']:
                            rl_conn.append(len(int_dict['routes_label']))
                    routes[n] = {'route': route, 'ends': [], 'ends_p': [], 'rl_conn': rl_conn}
            else:
                labels[labels==n] = 0
        return routes, labels, ends_rf

    def walkFaster(self, skel, start):

        route = [(int(start[1]), int(start[0]))]
        end = False
        while not end:
            end = True
            act = route[-1]
            skel[act[0], act[1]] = 0.
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                if skel[act[0] + dx, act[1] + dy]:
                    aim_x = act[0] + dx
                    aim_y = act[1] + dy
                    route.append((aim_x, aim_y))
                    end = False
                    break

        route = np.array(route)
        route -= 1

        return route

        # curr_pixel = [start[1], start[0]]
        # path = [curr_pixel]
        # while True:
        #     x, y = curr_pixel
        #     skel[curr_pixel[0], curr_pixel[1]] = 0.
        #     c_x, c_y = np.where(skel[x-1:x+2, y-1:y+2])
        #     c_x += x-1
        #     c_y += y-1
        #     num_neighbours = len(c_x)
        #     if num_neighbours == 1:
        #         curr_pixel = [c_x[0], c_y[0]]
        #         path.append(curr_pixel)
        #     elif num_neighbours == 0:
        #         break
        #     else:
        #         print('Bug')
        #         break
        #
        # path = np.array(path)
        # path -= 1
        #
        # return path

    def extractEndslist(self, skel):
        ends = self.extractEnds(skel)
        for e in ends:
            if e.shape[0] == 0:
                return []

        return list(zip(ends[1], ends[0]))

    def extractEnds(self, skel):

        skel = skel.copy()
        skel[skel != 0] = 1
        skel = np.uint8(skel)

        kernel = np.uint8([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])

        src_depth = -1
        filtered = cv2.filter2D(skel, src_depth, kernel)

        p_ends = np.where(filtered == 11)

        return np.array([p_ends[0], p_ends[1]])

    def extractInts(self, skel):

        skel = skel.copy()
        skel[skel != 0] = 1
        skel = np.uint8(skel)

        kernel = np.uint8([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])

        src_depth = -1
        filtered = cv2.filter2D(skel, src_depth, kernel)

        p_ints = np.where(filtered > 12)

        return np.array([p_ints[0], p_ints[1]])

    def distance2D(self, point1, point2):
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def distance3D(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5

    def estimateRoutewidthFromSegment(self, routes, dist_img, min_px = 3):
        for i in routes.keys():
            widths = [round(dist_img[tuple(p)]) for p in routes[i]['route']]
            # widths_int = [np.round(dist_img[tuple(p)]) for p in routes[i]['route']]
            avg_width = np.mean(widths)
            max_width = np.max(widths)
            routes[i]['width'] = (avg_width, max_width)
        self.total_mean_width = np.mean([max(routes[i]['width'][0], self.kernel_size) for i in routes.keys()])
        return routes

    def prune_short_routes(self, routes, ends, labels, skel, IMG_W=672, IMG_H=896):
        del_list = []
        for i in routes.keys():
            len_route = len(routes[i]['route'])
            rl_conn = routes[i]['rl_conn']
            if len_route < self.kernel_size * 2:
                del_list.append(i)
            else:
                if len(rl_conn) >= 2 and max(rl_conn) <= 3:
                    if len_route < max(routes[i]['width'][1], self.kernel_size * 2) * 3:
                        del_list.append(i)

        for del_index in del_list:
            p1 = (routes[del_index]['route'][0][1], routes[del_index]['route'][0][0])
            p2 = (routes[del_index]['route'][-1][1], routes[del_index]['route'][-1][0])
            ends = list(filter(lambda x: (x != p1 and x != p2), ends))
            for row, col in routes[del_index]['route']:
                labels[row, col] = 0
                skel[row, col] = 0
            del routes[del_index]
        return routes, ends, labels, skel

    def prune_2Y_routes(self, routes, ends_dict_rf, ints_dict_rf):
        del_list = []
        for i in routes.keys():
            if ends_dict_rf[routes[i]['ends'][0]]['int_label'] != -1 and ends_dict_rf[routes[i]['ends'][1]]['int_label'] != -1:
                if len(ints_dict_rf[ends_dict_rf[routes[i]['ends'][0]]['int_label']]['int_ends']) == 3 and \
                    len(ints_dict_rf[ends_dict_rf[routes[i]['ends'][1]]['int_label']]['int_ends']) == 3:
                    del_list.append(i)
        for del_index in del_list:
            del routes[del_index]
        return routes


    def constructIntsDict(self, skel, labels, dist_img, ints_dict_rf):
        skel_img = skel.copy()
        for i, int_dict in ints_dict_rf.items():
            point = int_dict['point']
            int_dict['int_radius'] = dist_img[point[0]][point[1]]
            cv2.circle(labels, tuple([point[1], point[0]]), round(int_dict['int_radius']), 0, -1)
            cv2.circle(skel_img, tuple([point[1], point[0]]), round(int_dict['int_radius']), 0, -1)
        return skel_img, labels, ints_dict_rf


    def constructEndsDict(self, routes, routes_im, source_img, ends, ints_dict):
        ends_dict = {}
        if ends.shape[0] > 0:
            ends_list = list(zip(ends[0], ends[1]))
        else:
            ends_list = []
        for i, end in enumerate(ends_list):
            end_type = 'iso'
            int_label = -1
            for j, int_dict in ints_dict.items():
                if self.distance2D(end, int_dict['point']) < int_dict['int_radius'] + 2 * self.kernel_size:
                    ints_dict[j]['int_ends'].append(i)
                    int_label = j
                    end_type = 'int'
            ends_dict[i] = {"point": end,
                            "route_label": routes_im[end[0]][end[1]],
                            "point_label": i,
                            "point_type": end_type,
                            "pair_ends": [],
                            "int_label": int_label,
                            "end_radius": routes[routes_im[end[0]][end[1]]]['width'][0],
                            "end_hsv": source_img[end[0]][end[1]]}
            routes[routes_im[end[0]][end[1]]]['ends'].append(i)
        return ends_dict, ints_dict

    def constructEndsDict_New(self, routes, routes_im, source_img, ends, ints_dict):
        ends_dict = {}
        for i, end in enumerate(ends):
            end = (end[1], end[0])
            ends_dict[i] = {"point": end,
                            "route_label": routes_im[end[0]][end[1]],
                            "point_label": i,
                            "point_type": 'iso',
                            "pair_ends": [],
                            "end_radius": routes[routes_im[end[0]][end[1]]]['width'][0],
                            "end_hsv": source_img[end[0]][end[1]]}
            routes[routes_im[end[0]][end[1]]]['ends'].append(i)
            routes[routes_im[end[0]][end[1]]]['ends_p'].append(end)
        for j, int_dict in ints_dict.items():
            p0 = int_dict['point']
            routes_label = int_dict['routes_label']
            for route_label in routes_label:
                try:
                    p1 = routes[route_label]['ends_p'][0]
                    p2 = routes[route_label]['ends_p'][1]
                    dist_p1 = self.distance2D(p0, p1)
                    dist_p2 = self.distance2D(p0, p2)
                    if dist_p1 < int_dict['int_radius'] + 2 * self.kernel_size:
                        ends_dict[routes[route_label]['ends'][0]]['point_type'] = 'int'
                        int_dict['int_ends'].append(routes[route_label]['ends'][0])
                    if dist_p2 < int_dict['int_radius'] + 2 * self.kernel_size:
                        ends_dict[routes[route_label]['ends'][1]]['point_type'] = 'int'
                        int_dict['int_ends'].append(routes[route_label]['ends'][1])
                except:
                    continue
        return ends_dict, ints_dict

    # 对于任意非交叉点的端点:
    # [1].是否处于图像边缘,是,则停止检测;
    # [2].计算该顶点的方向,归属到米字型的八个方向;
    # [3].在该方向对应的90度范围内寻找对应的
    def checkForContinuity(self, hsv, skel, ends_dict_rf, dir_ends_dict):
        dir_coor_dict = {'l': ['ru', 'r', 'rb'],
                         'lu': ['r', 'rb', 'b'],
                         'u': ['rb', 'b', 'lb'],
                         'ru': ['b', 'lb', 'l'],
                         'r': ['lb', 'l', 'lu'],
                         'rb': ['l', 'lu', 'u'],
                         'b': ['lu', 'u', 'ru'],
                         'lb': ['u', 'ru', 'r']}
        have_paired = []
        end_pairs = []
        for i, end_dict in ends_dict_rf.items():
            if end_dict in have_paired or end_dict['point_type'] != 'iso':
                continue
            end_border = end_dict['border']
            if end_border:
                continue
            end_dir = end_dict['dir_c']
            coor_dir_list = dir_coor_dict[end_dir]
            wait_list = []
            for coor_dir in coor_dir_list:
                wait_list += dir_ends_dict[coor_dir]
            for end_dict_ in wait_list:
                if end_dict_ in have_paired or end_dict['point_type'] != 'iso':
                    continue
                if end_dict_['border'] or end_dict['route_label'] == end_dict_['route_label']:
                    continue
                CM = self.calcEndSimilarity(end_dict, end_dict_, skel)
                if CM < 0.75:
                    # ================================
                    p1 = end_dict['point']
                    p2 = end_dict_['point']
                    path_x = list(
                        map(int, list(np.linspace(p1[0], p2[0], num=10))))
                    path_y = list(
                        map(int, list(np.linspace(p1[1], p2[1], num=10))))
                    hsv_path = hsv[path_x, path_y]
                    avg_hsv = hsv[path_x, path_y].mean(axis=0)
                    diff_hsv = hsv[path_x, path_y].std(axis=0).sum() / len(path_x)
                    if self.if_debug:
                        back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        cv2.line(back, (p1[1], p1[0]), (p2[1], p2[0]), (0, 0, 0))
                        cv2.imwrite('data/debug_results/conn_test/path_conn_{}.png'.format(diff_hsv), back)
                    # ================================
                    if diff_hsv > 5:
                        continue

                    end_dict['point_type'] = 'iso_p'
                    end_dict_['point_type'] = 'iso_p'
                    end_pairs.append((-1, end_dict['point_label'], end_dict_['point_label']))
                    have_paired.append(end_dict)
                    have_paired.append(end_dict_)
        return end_pairs

    def mergeEnds(self, skel, ends_dict_rf, end_pairs):
        skel_ = skel.copy()
        for end_pair in end_pairs:
            end_dict_1 = ends_dict_rf[end_pair[0]]
            end_dict_2 = ends_dict_rf[end_pair[1]]
            end_p1 = end_dict_1['point']
            end_p2 = end_dict_2['point']
            cv2.line(skel_, (end_p1[1], end_p1[0]), (end_p2[1], end_p2[0]), 255, thickness=1)
        return skel_

    def floodfill(self, window):
        height, width = window.shape
        center = (width//2, height//2)
        if window[center[1], center[0]] != 0:
            mask = np.zeros((height+2, width+2), dtype=np.uint8)
            cv2.floodFill(window, mask, center, 255, loDiff=1, upDiff=1, flags=8|cv2.FLOODFILL_MASK_ONLY)
            mask = mask[1:-1, 1:-1]
            window[mask==0]=0
        return window

    def calcRouteDirection(self, routes_im, ends_dict_rf, type='iso'):
        IMG_W, IMG_H = routes_im.shape[1], routes_im.shape[0]
        Dir_L = {'l': 'r', 'lu': 'rb', 'u': 'b', 'ru': 'lb', 'r': 'l', 'rb': 'lu', 'b': 'u', 'lb': 'ru'}
        dir_ends_dict = {'l': [], 'lu': [], 'u': [], 'ru': [], 'r': [], 'rb': [], 'b': [], 'lb': []}
        for i, end_dict in ends_dict_rf.items():
            if end_dict['point_type'] != type:
                continue
            end_point = end_dict['point']
            end_route_label = end_dict['route_label']
            if end_point[1] < (4 * self.kernel_size) or end_point[1] > (IMG_W - 4 * self.kernel_size - 1) or end_point[0] < (4 * self.kernel_size) or end_point[0] > (IMG_H - 4 * self.kernel_size - 1):
                end_dict['border'] = True
            else:
                end_dict['border'] = False
            end_window = routes_im[max(0, (end_point[0] - 2 * self.kernel_size)):min(IMG_H, (end_point[0] + 2 * self.kernel_size)),
                                   max(0, (end_point[1] - 2 * self.kernel_size)):min(IMG_W, (end_point[1] + 2 * self.kernel_size))]
            filled_window = end_window.copy()
            filled_window = self.floodfill(filled_window)
            EWRP = np.where(filled_window == end_route_label)
            # if self.if_debug:
            #     saved_window = cv2.resize(end_window, (256, 256))
            #     saved_window[saved_window > 0] = 255
            #     saved_window = ~saved_window
            #     cv2.imwrite('data/debug_results/similarity_test/end_window.png', saved_window)
            EWRP -= np.ones_like(EWRP) * (2 * self.kernel_size)
            y_sum = np.sum(EWRP[0])
            x_sum = np.sum(EWRP[1])
            if x_sum == 0:
                if y_sum > 0:
                    dir_n = 'u'
                elif y_sum < 0:
                    dir_n = 'b'
            elif x_sum > 0:
                if y_sum > 2.4 * abs(x_sum):
                    dir_n = 'u'
                elif y_sum > 0.4 * abs(x_sum) and y_sum <= 2.4 * abs(x_sum):
                    dir_n = 'ru'
                elif y_sum > -0.4 * abs(x_sum) and y_sum <= 0.4 * abs(x_sum):
                    dir_n = 'r'
                elif y_sum > -2.4 * abs(x_sum) and y_sum <= -0.4 * abs(x_sum):
                    dir_n = 'rb'
                else:
                    dir_n = 'b'
            else:
                if y_sum > 2.4 * abs(x_sum):
                    dir_n = 'u'
                elif y_sum > 0.4 * abs(x_sum) and y_sum <= 2.4 * abs(x_sum):
                    dir_n = 'lu'
                elif y_sum > -0.4 * abs(x_sum) and y_sum <= 0.4 * abs(x_sum):
                    dir_n = 'l'
                elif y_sum > -2.4 * abs(x_sum) and y_sum <= -0.4 * abs(x_sum):
                    dir_n = 'lb'
                else:
                    dir_n = 'b'

            end_dict['dir_c'] = Dir_L[dir_n]
            end_dict['dir'] = (x_sum, y_sum)/np.linalg.norm((x_sum, y_sum))

            dir_ends_dict[Dir_L[dir_n]].append(end_dict)

        return ends_dict_rf, dir_ends_dict

    def calcEndSimilarity(self, end_dict_1, end_dict_2, skel, flag='end_pair'):
        IMG_H, IMG_W = skel.shape[0], skel.shape[1]
        point1 = end_dict_1['point']
        point2 = end_dict_2['point']
        dir_p1 = end_dict_1['dir']
        dir_p2 = end_dict_2['dir']
        width_p1 = end_dict_1['end_radius']
        width_p2 = end_dict_2['end_radius']
        hsv_p1 = end_dict_1['end_hsv']
        hsv_p2 = end_dict_2['end_hsv']
        if flag == 'end_pair':
            lambda_CE = 1
            lambda_CD = 1
            lambda_CC = 1
            lambda_CW = 0
            lambda_CH = 0.1
        elif flag == 'int_pair':
            lambda_CE = 0
            lambda_CD = 1
            lambda_CC = 1
            lambda_CW = 0
            lambda_CH = 1
        else:
            print("unknown type")
        CE = self.costEuclidean(point1, point2, IMG_H, IMG_W)
        CD = self.costDirection(dir_p1, dir_p2)
        CC = self.costCurvature(point1, point2, dir_p1, dir_p2)
        CW = self.costWidth(width_p1, width_p2)
        CH = self.costHSV(hsv_p1, hsv_p2)
        CM = lambda_CE * CE + lambda_CD * CD + lambda_CC * CC + lambda_CW * CW + lambda_CH * CH
        # print(CE, CD, CC, CW, CH, CM)
        if self.if_debug:
            back = cv2.cvtColor(skel * 255, cv2.COLOR_GRAY2BGR)
            back = ~back
            cv2.circle(back, (point1[1], point1[0]), 3, (0, 255, 0))
            cv2.circle(back, (point2[1], point2[0]), 3, (0, 255, 0))
            if flag == 'int_pair':
                back = back[max(0, ((point1[0] + point2[0])//2 - 20*self.kernel_size)):min(IMG_H, ((point1[0] + point2[0])//2 + 20*self.kernel_size)),
                            max(0, ((point1[1] + point2[1])//2 - 20*self.kernel_size)):min(IMG_W, ((point1[1] + point2[1])//2 + 20*self.kernel_size))]
            cv2.imwrite('data/debug_results/similarity_test/calcSimilarity_{}.jpg'.format(CM), back)
        return CM

    def costEuclidean(self, point1, point2, IMG_H, IMG_W):
        return (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5) / (
                    (IMG_H ** 2 + IMG_W ** 2) ** 0.5)

    def costDirection(self, dir1, dir2):
        vec1 = np.array(dir1)
        vec2 = np.array(dir2)
        return (1 - np.dot(-vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))) / 2

    def costCurvature(self, point1, point2, dir1, dir2):
        vec0 = np.array([(point2[1] - point1[1]), (point2[0] - point1[0])])
        vec1 = np.array(dir1)
        vec2 = np.array(dir2)
        CC1 = np.dot(vec0, -vec1) / (np.linalg.norm(vec0) * np.linalg.norm(vec1))
        CC2 = np.dot(-vec0, -vec2) / (np.linalg.norm(vec0) * np.linalg.norm(vec2))
        return (1 - min([CC1, CC2])) / 2

    def costWidth(self, width1, width2):
        return abs((width1 - width2) / (width1 + width2))

    def costHSV(self, hsv1, hsv2):
        hue_1, hue_2 = 2 * int(hsv1[0]), 2 * int(hsv2[0])
        # if min(hue_1, hue_2) < 10 and max(hue_1, hue_2) > 156:
        #     if hue_2 > hue_1:
        #         hue_2 -= 180
        #     else:
        #         hue_1 -= 180
        hue_1, hue_2 = hue_1 / 180 * math.pi, hue_2 / 180 * math.pi
        sat_1, sat_2 = int(hsv1[1]) / 255, int(hsv2[1]) / 255
        val_1, val_2 = int(hsv1[2]) / 255, int(hsv2[2]) / 255
        hsv_sp1 = [val_1, sat_1 * math.cos(hue_1), sat_1 * math.sin(hue_1)]
        hsv_sp2 = [val_2, sat_2 * math.cos(hue_2), sat_2 * math.sin(hue_2)]
        dis_hsv = self.distance3D(hsv_sp1, hsv_sp2)
        return dis_hsv / 2

    def handleIntersections(self, skel, ends_dict_rf, ints_dict_m):
        end_pairs = []
        to_delete = []
        for i, int_dict in ints_dict_m.items():
            int_end_num = len(int_dict['int_ends'])
            if int_end_num == 1:
                ends_dict_rf[int_dict['int_ends'][0]]['point_type'] = 'iso'
                to_delete.append(i)
            elif int_end_num == 2:
                end_pairs += self.handleTwo(skel, ends_dict_rf, i, int_dict)
                to_delete.append(i)
            elif int_end_num == 3:
                end_pairs += self.handleFork(skel, ends_dict_rf, i, int_dict)
            elif int_end_num == 4:
                end_pairs += self.handleCross(skel, ends_dict_rf, i, int_dict)
            else:
                end_pairs += self.handleNCross(skel, ends_dict_rf, i, int_dict)
        for k in to_delete:
            del ints_dict_m[k]
        return end_pairs

    def handleTwo(self, skel, ends_dict_rf, i, int_dict):
        # print('handleTwo')
        end_pairs = []
        int_dict_ends = int_dict['int_ends']
        CM_01 = self.calcEndSimilarity(ends_dict_rf[int_dict_ends[0]], ends_dict_rf[int_dict_ends[1]], skel, 'int_pair')
        if CM_01 < 0.75:
            end_pairs.append((i, int_dict_ends[0], int_dict_ends[1]))
        else:
            ends_dict_rf[int_dict_ends[0]]['point_type'] = 'iso'
            ends_dict_rf[int_dict_ends[1]]['point_type'] = 'iso'
        return end_pairs

    def handleFork(self, skel, ends_dict_rf, i, int_dict):
        # print('handleFork')
        end_pairs = []
        int_dict_ends = int_dict['int_ends']
        CM_01 = self.calcEndSimilarity(ends_dict_rf[int_dict_ends[0]], ends_dict_rf[int_dict_ends[1]], skel, 'int_pair')
        CM_02 = self.calcEndSimilarity(ends_dict_rf[int_dict_ends[0]], ends_dict_rf[int_dict_ends[2]], skel, 'int_pair')
        CM_12 = self.calcEndSimilarity(ends_dict_rf[int_dict_ends[1]], ends_dict_rf[int_dict_ends[2]], skel, 'int_pair')

        index_l = ['01', '02', '12']
        CM_l = [CM_01, CM_02, CM_12]
        CM_min_index = index_l[CM_l.index(min(CM_l))]
        CM_min = min(CM_l)
        ind1 = int(CM_min_index[0])
        ind2 = int(CM_min_index[1])
        end_pairs.append((i, int_dict_ends[ind1], int_dict_ends[ind2]))
        index_l.remove(CM_min_index)
        CM_l.remove(min(CM_l))
        if min(CM_l) < max(7 * CM_min, 0.75):
            CM_min_index = index_l[CM_l.index(min(CM_l))]
            ind1 = int(CM_min_index[0])
            ind2 = int(CM_min_index[1])
            end_pairs.append((i, int_dict_ends[ind1], int_dict_ends[ind2]))
        else:
            index_s_l = ['0', '1', '2']
            index_left = int(list(set(index_s_l) - set(list(CM_min_index)))[0])
            ends_dict_rf[int_dict_ends[index_left]]['point_type'] = 'iso'
            ends_dict_rf[int_dict_ends[index_left]]['int_label'] = -1
        return end_pairs

    def handleCross(self, skel, ends_dict_rf, i, int_dict):
        # print('handleCross')
        end_pairs = []
        int_dict_ends = int_dict['int_ends']
        index_l = ['01', '02', '03', '12', '13', '23']
        CM = {}
        for index in index_l:
            ind1 = int(index[0])
            ind2 = int(index[1])
            CM[index] = self.calcEndSimilarity(ends_dict_rf[int_dict_ends[ind1]], ends_dict_rf[int_dict_ends[ind2]], skel, 'int_pair')
        # 找到四个点匹配过程中的最小值和次小值,返回其index
        # CM_l = [v for v in CM.values()]
        # CM_min_index = index_l[CM_l.index(min(CM_l))]
        # index_s_l = ['0', '1', '2', '3']                       # 该情况下可被认为是交叉路口
        # index_left_l = list(set(index_s_l) - set(list(CM_min_index)))
        # end_pairs.append((int_dict_ends[int(CM_min_index[0])], int_dict_ends[int(CM_min_index[1])]))
        # end_pairs.append((int_dict_ends[int(index_left_l[0])], int_dict_ends[int(index_left_l[1])]))
        # 找到一个两两组合使得其最小
        index_pair_l = [['01', '23'], ['02', '13'], ['03', '12']]
        CM_pair_l = [CM[index_pair[0]] + CM[index_pair[1]] for index_pair in index_pair_l]
        CM_pair_min = index_pair_l[CM_pair_l.index(min(CM_pair_l))]
        for index in CM_pair_min:
            end_pairs.append((i, int_dict_ends[int(index[0])], int_dict_ends[int(index[1])]))
        return end_pairs

    def repeatIndex(self, index1, index2):
        for ch in index2:
            if ch in index1:
                return ch
        return ''

    def handleNCross(self, skel, ends_dict_rf, i, int_dict):
        # print('handleNCross')
        end_pairs = []
        int_dict_ends = int_dict['int_ends']
        cross_num = len(int_dict_ends)
        index_all = []
        for i in range(cross_num - 1):
            for j in range(i + 1, cross_num):
                index_all.append(str(i) + str(j))
        CM = {}
        end = []
        index_l = []
        for index in index_all:
            ind1 = int(index[0])
            ind2 = int(index[1])
            simi = self.calcEndSimilarity(ends_dict_rf[int_dict_ends[ind1]], ends_dict_rf[int_dict_ends[ind2]],
                                          skel, 'int_pair')
            if not math.isnan(simi):
                index_l.append(index)
                CM[index] = simi
                if str(ind1) not in end:
                    end.append(str(ind1))
                if str(ind2) not in end:
                    end.append(str(ind2))
        if len(end) < cross_num:
            for i in range(cross_num):
                if str(i) not in end:
                    ends_dict_rf[int_dict_ends[i]]['point_type'] = 'iso'
        CM_l = [v for v in CM.values()]
        end_already_in = []
        while end_already_in != end:
            CM_min_index = index_l[CM_l.index(min(CM_l))]
            index_l.remove(CM_min_index)
            CM_l.remove(min(CM_l))
            end_pairs.append((i, int_dict_ends[int(CM_min_index[0])], int_dict_ends[int(CM_min_index[1])]))
            if CM_min_index[0] not in end_already_in:
                end_already_in.append(CM_min_index[0])
            if CM_min_index[1] not in end_already_in:
                end_already_in.append(CM_min_index[1])
            end_already_in.sort()
        return end_pairs

    def cleanEndPairs(self, end_pairs):
        end_pairs_clean = []
        for end_pair in end_pairs:
            if end_pair not in end_pairs_clean:
                end_pairs_clean.append(end_pair)
        return end_pairs_clean

    def mergeIntsFromRoutes(self, routes, ints_dict):
        if len(ints_dict) == 0:
            return ints_dict

        labelsw = [v["routes_label"] for k, v in ints_dict.items()]
        labels = [item for sublist in labelsw for item in sublist]
        labels = np.unique(labels)
        route_keys = list(routes.keys())
        diff = set(labels).difference(set(route_keys))

        to_delete = []
        ints_dict_new = {}
        for dkey in diff:
            new_routes = []
            new_ends = []
            x, y = [], []
            keys_to_delete = []
            for k, v in ints_dict.items():
                if dkey in v["routes_label"]:
                    new_routes.extend(v["routes_label"])
                    new_ends.extend(v["int_ends"])
                    x.append(v["point"][0])
                    y.append(v["point"][1])
                    keys_to_delete.append(k)
            routes = np.unique(new_routes)
            routes = [s for s in routes if s != dkey]
            ends = np.unique(new_ends)
            point = tuple([np.int_(np.mean(x)), np.int_(np.mean(y))])
            ints_dict[keys_to_delete[0]]['point'] = point
            ints_dict[keys_to_delete[0]]['int_ends'] = ends
            ints_dict[keys_to_delete[0]]['routes_label'] = routes
            to_delete.extend(keys_to_delete[1:])

        for d in ints_dict.keys():
            if d not in to_delete:
                ints_dict_new[d] = ints_dict[d]
        return ints_dict_new