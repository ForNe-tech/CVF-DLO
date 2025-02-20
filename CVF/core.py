import collections
import itertools
from matplotlib import cm
import cv2
import numpy as np
import arrow
import math

from .segment.predict import SegNet
from .process.skeletonize_EWD_color_Final import Skeletonize as Skeletonize_EWD
from .process.skeletonize_LAB import Skeletonize as Skeletonize_LAB
from .process.skeletonize_SBHC_Fast import Skeletonize as Skeletonize_SBHC
from .draw_drac import calcEllipseFromEnds, draw_drac

from time import time

class Pipeline():

    def __init__(self, checkpoint_seg=None, colors=None, img_w=1280, img_h=960, if_debug=True, scene='LAB'):
        if checkpoint_seg is not None:
            self.network_seg = SegNet(model_name="deeplabv3plus_resnet50", checkpoint_path=checkpoint_seg, img_w=img_w, img_h=img_h)
        else:
            self.network_seg = None
        self.if_debug = if_debug
        if colors is None:
            self.cmap = self.voc_cmap(N=256, normalized=False)
        else:
            self.cmap = colors
        self.scene = scene
        self.IMG_H = img_h
        self.IMG_W = img_w
        self.dist_img = None

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
        return cmap.tolist()

    def run(self, source_img, mask_img=None, mask_th=127, verbose=False):

        # =========================================================================
        verbose_print = print if verbose else lambda x: None
        times = []
        times.append(time())
        # =========================================================================

        if self.network_seg is not None and mask_img is None:
            mask_img = self.network_seg.predict_img(source_img)
        elif self.network_seg is None and mask_img is None:
            print('Error: mask_img is not found!')
        else:
            print('Get mask_img from DIR')

        mask_img[mask_img < mask_th] = 0
        mask_img[mask_img != 0] = 255

        img_Ex = np.zeros((source_img.shape[0] + 10, source_img.shape[1] + 10, source_img.shape[2]), dtype=np.uint8)
        mask_Ex = np.zeros((mask_img.shape[0] + 10, mask_img.shape[1] + 10), dtype=np.uint8)
        img_Ex[5:-5, 5:-5] = source_img
        mask_Ex[5:-5, 5:-5] = mask_img

        # kernel = np.ones((5, 5), np.uint8)
        # mask_img_G = cv2.morphologyEx(mask_Ex, cv2.MORPH_CLOSE, kernel)

        mask_img_G = mask_Ex.copy()

        if self.if_debug:
            cv2.imwrite('data/debug_results/mask_guass.png', mask_img_G)

        # print(mask_img_G.sum()/255)

        img_out, skeleton_or_routelist, times, routes, end_pairs, ends_dict = self.process(source_img=img_Ex, mask_img=mask_img_G, verbose=verbose, times=times)

        # skel_thick = cv2.dilate(skeleton_or_routelist, (5, 5), iterations=3)
        # mask_img[skel_thick != 0] = 255

        img_out = img_out[5:-5, 5:-5]

        return img_out, mask_img, skeleton_or_routelist, times, routes, end_pairs, ends_dict


    def process(self, source_img, mask_img, verbose=False, times=None):

        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2HSV)

        if self.scene == 'EWD':
            SKEL = Skeletonize_EWD(if_debug=self.if_debug)
        elif self.scene == 'LAB':
            SKEL = Skeletonize_LAB(if_debug=self.if_debug)
        elif self.scene == 'SBHC':
            SKEL = Skeletonize_SBHC(if_debug=self.if_debug)

        # =========================================================================
        verbose_print = print if verbose else lambda x: None
        s = times.pop()
        times.append(time() - s)
        verbose_print("Pre Process time: {:.5f}".format(times[-1]))
        times.append(time())
        # =========================================================================

        skeleton_rf, routes, end_pairs, ends_dict_rf, int_dicts_m, times = SKEL.run(mask_img, source_img=source_img, verbose=verbose, times=times)

        # end_pairs_dict = self.calcEndPairsContinuity(ends_dict_rf, end_pairs, source_img)

        skeleton_m = self.mergeEnds(skeleton_rf, ends_dict_rf, end_pairs)
        ends_dict_rf = self.combineEndsDict(routes, ends_dict_rf, end_pairs)

        if self.scene == 'EWD':
            combEnds_list = self.combineEndsRoutes_New(routes, ends_dict_rf)
            combEnds_list = self.deleteCloseRoutes(routes, ends_dict_rf, combEnds_list)
            combEnds_list = self.validFromHSV_Finetune(routes, ends_dict_rf, combEnds_list, source_img)

            # =========================================================================
            s = times.pop()
            times.append(time() - s)
            verbose_print("Path generation time: {:.5f}".format(times[-1]))
            s = time()
            # =========================================================================

            route_seg = self.combEndsToRoutes(source_img, routes, ends_dict_rf, int_dicts_m, end_pairs, combEnds_list)
            # route_seg = self.showCombRoutes(skeleton_m, routes, ends_dict_rf, combEnds_list)
            # route_seg = self.showCombRoutes_WithCrossOrder(skeleton_m, routes, ends_dict_rf, combEnds_list, int_dicts_m, source_img)

            # =========================================================================
            times.append(time() - s)
            verbose_print("Plotting time: {:.5f}".format(times[-1]))
            verbose_print("Total time: {:.5f}".format(sum(times)))

            return route_seg, skeleton_m, times, routes, end_pairs, ends_dict_rf

        elif self.scene == 'LAB':
            combEnds_list = self.combineEndsRoutes_NoCircle(routes, ends_dict_rf)
            combEnds_list = self.deleteCloseRoutes(routes, ends_dict_rf, combEnds_list)
            route_seg = self.showCombRoutes(skeleton_m, routes, ends_dict_rf, combEnds_list)
            route_seg_list = self.showCombRoutes_Single(skeleton_m, routes, ends_dict_rf, combEnds_list)

            verbose_print("Path generation time: {:.5f}".format(time() - s))

            return route_seg, route_seg_list, times, routes, end_pairs, ends_dict_rf

        elif self.scene == 'SBHC':
            combEnds_list = self.combineEndsRoutes_New(routes, ends_dict_rf)
            combEnds_list = self.deleteCloseRoutes(routes, ends_dict_rf, combEnds_list)
            combEnds_list = self.validFromHSV_Finetune(routes, ends_dict_rf, combEnds_list, source_img)

            # =========================================================================
            s = times.pop()
            times.append(time() - s)
            verbose_print("Path generation time: {:.5f}".format(times[-1]))
            s = time()
            # =========================================================================

            route_seg = self.combEndsToRoutes(source_img, routes, ends_dict_rf, int_dicts_m, end_pairs, combEnds_list)
            # route_seg = self.showCombRoutes(skeleton_m, routes, ends_dict_rf, combEnds_list)
            # route_seg = self.showCombRoutes_WithCrossOrder(skeleton_m, routes, ends_dict_rf, combEnds_list, int_dicts_m, source_img, dist_img, mask_img_origin)

            route_seg = cv2.dilate(route_seg, (7, 7))

            # =========================================================================
            times.append(time() - s)
            verbose_print("Plotting time: {:.5f}".format(times[-1]))
            verbose_print("Total time: {:.5f}".format(sum(times)))

            return route_seg, skeleton_m, times, routes, end_pairs, ends_dict_rf



    def calcEndPairsContinuity(self, ends_dict, end_pairs, hsv):
        end_pairs_dict = {}
        for end_pair in end_pairs:
            end_pair_continuity = self.calcCrossContinuity(end_pair, ends_dict, hsv)
            end_pair_ = [end_pair[0], end_pair[1]]
            end_pair = tuple(sorted(end_pair_))
            end_pairs_dict[end_pair] = end_pair_continuity
        return end_pairs_dict

    def mergeEnds(self, skel, ends_dict_rf, end_pairs):
        skel_ = skel.copy()
        for end_pair in end_pairs:
            end_dict_1 = ends_dict_rf[end_pair[1]]
            end_dict_2 = ends_dict_rf[end_pair[2]]
            end_p1 = end_dict_1['point']
            end_p2 = end_dict_2['point']
            cv2.line(skel_, (end_p1[1], end_p1[0]), (end_p2[1], end_p2[0]), 255, thickness=1)
        return skel_

    def combineEndsDict(self, routes, ends_dict_rf, end_pairs):
        for i, end_dict in ends_dict_rf.items():
            for end_pair in end_pairs:
                if end_dict['point_label'] == end_pair[1]:
                    end_dict['pair_ends'].append(int(end_pair[2]))
                elif end_dict['point_label'] == end_pair[2]:
                    end_dict['pair_ends'].append(int(end_pair[1]))
            route_label = end_dict['route_label']
            ends = routes[route_label]['ends'].copy()
            ends.remove(i)
            end_dict['route_end'] = ends[0]
        return ends_dict_rf


    def combineEndsRoutes(self, routes, ends_dict):
        combEnds_list = []
        for i, end_dict in ends_dict.items():
            # 从没有配对端点的孤立端点出发
            if end_dict['point_type'] == 'iso':
                first_label = end_dict['point_label']
                have_traversed = [first_label]

                def addMultiItemsList(baseList, addList):
                    large_baseList = []
                    for baselist in baseList:
                        for addlist in addList:
                            large_baseList.append(baselist + addlist)
                    return large_baseList

                def getRouteEnd(end_label):
                    next_label = ends_dict[end_label]['route_end']
                    if ends_dict[next_label]['point_type'] == 'iso':
                        have_traversed.append(next_label)
                        return [[next_label]]
                    else:
                        have_traversed.append(next_label)
                        return addMultiItemsList([[next_label]], getPairEnd(next_label))

                def getPairEnd(end_label):
                    next_labels = ends_dict[end_label]['pair_ends']
                    re_baseList = []
                    for next_label in next_labels:
                        if next_label not in have_traversed:
                            have_traversed.append(next_label)
                            re_baseList += addMultiItemsList([[next_label]], getRouteEnd(next_label))
                        else:
                            have_traversed.append(next_label)
                            re_baseList = [[next_label]]
                    return re_baseList

                combEnds_list += addMultiItemsList([[first_label]], getRouteEnd(first_label))
        return combEnds_list


    def combineEndsRoutes_New(self, routes, ends_dict):
        combEnds_list = []
        for i, end_dict in ends_dict.items():
            # 从没有配对端点的孤立端点出发
            if end_dict['point_type'] == 'iso':
                first_label = end_dict['point_label']
                have_traversed = []

                def addMultiItemsList(baseList, addList):
                    large_baseList = []
                    for baselist in baseList:
                        for addlist in addList:
                            large_baseList.append(baselist + addlist)
                    return large_baseList

                def getRouteEnd(end_label):
                    next_label = ends_dict[end_label]['route_end']
                    if ends_dict[next_label]['point_type'] == 'iso':
                        return [[next_label]]
                    else:
                        return addMultiItemsList([[next_label]], getPairEnd(next_label))

                def getPairEnd(end_label):
                    next_labels = ends_dict[end_label]['pair_ends']
                    re_baseList = []
                    for next_label in next_labels:
                        end_pair = tuple(sorted([end_label, next_label]))
                        if end_pair not in have_traversed:
                            have_traversed.append(end_pair)
                            re_baseList += addMultiItemsList([[next_label]], getRouteEnd(next_label))
                            have_traversed.remove(end_pair)
                    return re_baseList

                combEnds_list += addMultiItemsList([[first_label]], getRouteEnd(first_label))
        return combEnds_list

    def combineEndsRoutes_NoCircle(self, routes, ends_dict):
        combEnds_list = []
        for i, end_dict in ends_dict.items():
            # 从没有配对端点的孤立端点出发
            if end_dict['point_type'] == 'iso':
                first_label = end_dict['point_label']

                def addMultiItemsList(baseList, addList):
                    large_baseList = []
                    for baselist in baseList:
                        for addlist in addList:
                            large_baseList.append(baselist + addlist)
                    return large_baseList

                def getRouteEnd(end_label):
                    next_label = ends_dict[end_label]['route_end']
                    if ends_dict[next_label]['point_type'] == 'iso':
                        return [[next_label]]
                    else:
                        return addMultiItemsList([[next_label]], getPairEnd(next_label))

                def getPairEnd(end_label):
                    next_labels = ends_dict[end_label]['pair_ends']
                    re_baseList = []
                    for next_label in next_labels:
                        re_baseList += addMultiItemsList([[next_label]], getRouteEnd(next_label))
                    return re_baseList

                combEnds_list += addMultiItemsList([[first_label]], getRouteEnd(first_label))
        return combEnds_list

    def combineEndsRoutes_SingleLine(self, routes, ends_dict):
        ends_dict_temp = ends_dict.copy()
        combEnds_list = []
        for i, end_dict in ends_dict.items():
            # 从没有配对端点的孤立端点出发
            if end_dict['point_type'] == 'iso':
                first_label = end_dict['point_label']

                combEnds = [first_label]
                def get_next_label(last_label, flag):
                    if flag:    # flag为True时表示处理intersection,flag为Flase时表示处理path
                        if ends_dict_temp[last_label]['point_type'] == 'iso':
                            combEnds_list.append(combEnds)
                            return
                        else:
                            next_label = ends_dict_temp[last_label]['pair_ends'][0]
                            ends_dict_temp[last_label]['pair_ends'].remove(next_label)
                            combEnds.append(next_label)
                            flag = ~flag
                            get_next_label(next_label, flag)
                    else:
                        next_label = ends_dict_temp[last_label]['route_end']
                        combEnds.append(next_label)
                        flag = ~flag
                        get_next_label(next_label, flag)

                get_next_label(first_label, False)

        return combEnds_list

    def deleteCloseRoutes(self, routes, ends_dict, combEnds):
        del_list = []
        have_existed = []
        for i, singleRoute in enumerate(combEnds):
            k = 0
            total_len_route = 0
            while k < len(singleRoute):
                total_len_route += len(routes[ends_dict[singleRoute[k]]['route_label']]['route'])
                k += 2
            if total_len_route < 50:
                del_list.append(i)
                continue
            if singleRoute[0] > singleRoute[-1]:
                singleRoute.reverse()
            if ends_dict[singleRoute[-1]]['point_type'] != 'iso' or ends_dict[singleRoute[0]]['point_type'] != 'iso':
                del_list.append(i)
            else:
                if singleRoute in have_existed:
                    del_list.append(i)
                else:
                    have_existed.append(singleRoute)
        del_list.reverse()
        for del_index in del_list:
            del combEnds[del_index]
        return combEnds


    def validFromHSV(self, routes, ends_dict, combEnds, source_img):
        del_list = []
        start_end = {}
        for i, singleRoute in enumerate(combEnds):
            diff_ends = self.costHSV(ends_dict[singleRoute[0]]['end_hsv'], ends_dict[singleRoute[-1]]['end_hsv'])
            diff_route = self.calcRouteScore(routes, singleRoute, ends_dict, source_img)
            if singleRoute[0] not in start_end:
                start_end[singleRoute[0]] = [(i, diff_ends, diff_route)]
            else:
                start_end[singleRoute[0]].append((i, diff_ends, diff_route))
            if singleRoute[-1] not in start_end:
                start_end[singleRoute[-1]] = [(i, diff_ends, diff_route)]
            else:
                start_end[singleRoute[-1]].append((i, diff_ends, diff_route))
        for j, pot_routes in start_end.items():
            if len(pot_routes) == 1:
                continue
            thre = min([float(100 * v[1] + v[2]) for v in pot_routes])
            for pot_route in pot_routes:
                if 100 * pot_route[1] + pot_route[2] > thre:
                    if pot_route[0] not in del_list:
                        del_list.append(pot_route[0])
        del_list.sort()
        del_list.reverse()
        for del_index in del_list:
            del combEnds[del_index]
        return combEnds

    def validFromHSV_Finetune(self, routes, ends_dict, combEnds, source_img):
        del_list = []
        route_score_dict = {}
        exist_ends = []
        used_ends = []
        for i, singleRoute in enumerate(combEnds):
            diff_ends = self.costHSV(ends_dict[singleRoute[0]]['end_hsv'], ends_dict[singleRoute[-1]]['end_hsv'])
            diff_route = self.calcRouteScore(routes, singleRoute, ends_dict, source_img)
            # route_score_dict[i] = round(100 * diff_ends + diff_route, 3)
            route_score_dict[i] = 100 * diff_ends + diff_route + 0.01 / len(singleRoute)
            if singleRoute[0] not in exist_ends:
                exist_ends.append(singleRoute[0])
            if singleRoute[-1] not in exist_ends:
                exist_ends.append(singleRoute[-1])

        reserved_combEnds = []
        while len(route_score_dict) > 0:
            sorted_items = sorted(route_score_dict.items(), key=lambda item: item[1])
            reserved_idx = sorted_items[0][0]
            reserved_combEnds.append(combEnds[reserved_idx])
            used_ends.append(combEnds[reserved_idx][0])
            used_ends.append(combEnds[reserved_idx][-1])
            del route_score_dict[reserved_idx]
            del_list = []
            for label in route_score_dict.keys():
                if combEnds[label][0] in used_ends or combEnds[label][-1] in used_ends:
                    del_list.append(label)
            for del_label in del_list:
                del route_score_dict[del_label]

        return reserved_combEnds

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

    def distance3D(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2) ** 0.5

    def calcRouteScore(self, routes, singleRoute, ends_dict, rgb):
        if len(singleRoute) <= 2:
            return 100
        route_score = []
        k = 0
        hsv_mid_list = []
        while k < len(singleRoute) - 1:
            route = routes[ends_dict[singleRoute[k]]['route_label']]['route']
            len_route = len(route)
            p_mid = route[len_route//2]
            hsv_p_mid = rgb[p_mid[0]][p_mid[1]]
            # p1 = ends_dict[singleRoute[k]]['point']
            # p2 = ends_dict[singleRoute[k+1]]['point']
            # hsv_p1 = ends_dict[singleRoute[k]]['end_hsv']
            # hsv_p2 = ends_dict[singleRoute[k+1]]['end_hsv']
            # hsv_p12 = tuple([hsv_p1[0]/2 + hsv_p2[0]/2, hsv_p1[1]/2 + hsv_p2[1]/2, hsv_p1[2]/2 + hsv_p2[2]/2])
            if self.if_debug:
                back = cv2.cvtColor(rgb, cv2.COLOR_HSV2BGR)
                cv2.circle(back, (p_mid[1], p_mid[0]), 3, (0, 255, 0))
                cv2.imwrite('data/debug_results/route_test/calcRouteScore.jpg', back)
            hsv_mid_list.append(hsv_p_mid)
            k += 2
        for j in range(1, len(hsv_mid_list)-1):
            route_score.append(self.costHSV(hsv_mid_list[0], hsv_mid_list[j]))
        return np.mean(route_score)



    def showCombRoutes(self, skel, routes, ends_dict, combEnds_list):
        back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        back_w = np.zeros_like(back)
        color_index = 0
        for j, singleRoute in enumerate(combEnds_list):
            turn = True
            for k in range(len(singleRoute)-1):
                point_label = singleRoute[k]
                # p1 = ends_dict[point_label]['point']
                # cv2.circle(back_w, (p1[1], p1[0]), 3, (0, 255, 0))
                line_color = (int(self.cmap[color_index][0]),
                              int(self.cmap[color_index][1]),
                              int(self.cmap[color_index][2]))
                if turn:
                    turn = False
                    route_label = ends_dict[point_label]['route_label']
                    radius = routes[route_label]['width'][0]
                    for point in routes[route_label]['route']:
                        cv2.circle(back_w, (point[1], point[0]), int(radius) + 1, line_color, -1)
                        back_w[point[0]][point[1]] = self.cmap[color_index]
                else:
                    turn = True
                    next_label = singleRoute[k+1]
                    route_label = ends_dict[next_label]['route_label']
                    end_p1 = ends_dict[point_label]['point']
                    end_p2 = ends_dict[next_label]['point']
                    line_thickness = max(int(routes[route_label]['width'][0] * 2 - 2), 3)
                    line_color = (int(self.cmap[color_index][0]), int(self.cmap[color_index][1]), int(self.cmap[color_index][2]))
                    cv2.line(back_w, (end_p1[1], end_p1[0]), (end_p2[1], end_p2[0]), line_color, thickness=line_thickness)
            # cv2.imwrite('data/debug_results/route_test/route_comb_visib_{}.jpg'.format(color_index), back_w)
            color_index += 1
        # cv2.imwrite('data/debug_results/route_test/route_comb_visib.jpg', back_w)
        return back_w

    def showCombRoutes_WithCrossOrder(self, skel, routes, ends_dict, combEnds_list, ints_dict, rgb):
        route_pairs_have_drawn = {}
        cross_pairs_have_drawn = {}
        back = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        back_w = np.zeros_like(back)
        color_index = 0
        for j, singleRoute in enumerate(combEnds_list):
            turn = True
            wait_for_redraw = False
            for k in range(len(singleRoute)-1):
                line_color = (int(self.cmap[color_index][0]),
                              int(self.cmap[color_index][1]),
                              int(self.cmap[color_index][2]))
                if turn:
                    turn = False
                    route_pair_ = [singleRoute[k], singleRoute[k+1]]
                    route_pair = tuple(sorted(route_pair_))
                    route_continuity = self.calcRouteContinuity(routes, singleRoute, k, ends_dict, rgb)
                    self.drawRoute(route_pair, ends_dict, routes, line_color, back_w)
                    connected_cross_pair = []
                    if k >= 2 and k < len(singleRoute) - 2:
                        cross_pair_l_ = [singleRoute[k - 1], singleRoute[k]]
                        cross_pair_l = tuple(sorted(cross_pair_l_))
                        connected_cross_pair.append(cross_pair_l)
                        cross_pair_r_ = [singleRoute[k + 1], singleRoute[k + 2]]
                        cross_pair_r = tuple(sorted(cross_pair_r_))
                        connected_cross_pair.append(cross_pair_r)
                    elif k < 2 and len(singleRoute) >= 4:
                        cross_pair_r_ = [singleRoute[k + 1], singleRoute[k + 2]]
                        cross_pair_r = tuple(sorted(cross_pair_r_))
                        connected_cross_pair.append(cross_pair_r)
                    elif k >= len(singleRoute) - 2 and len(singleRoute) >= 4:
                        cross_pair_l_ = [singleRoute[k - 1], singleRoute[k]]
                        cross_pair_l = tuple(sorted(cross_pair_l_))
                        connected_cross_pair.append(cross_pair_l)
                    if route_pair not in route_pairs_have_drawn.keys():
                        route_pairs_have_drawn[route_pair] = {'score': route_continuity,
                                                              'line_color': line_color,
                                                              'connected_pairs': connected_cross_pair}
                    else:
                        if route_pairs_have_drawn[route_pair]['score'] > route_continuity:  # 原有路线不如新路线,绘制并覆盖原有路线
                            self.drawRoute(route_pair, ends_dict, routes, line_color, back_w)
                            route_pairs_have_drawn[route_pair] = {'score': route_continuity,
                                                                  'line_color': line_color,
                                                                  'connected_pairs': connected_cross_pair}
                        else:
                            wait_for_redraw = True


                else:
                    turn = True
                    cross_pair_ = [singleRoute[k], singleRoute[k + 1]]
                    cross_pair = tuple(sorted(cross_pair_))
                    # 先画一条
                    self.drawCross(cross_pair, ends_dict, routes, line_color, back_w)
                    # 判断
                    cross_continuity = self.calcCrossContinuity(cross_pair, ends_dict, rgb)
                    intersect_pairs = self.CrossExist(cross_pair, cross_pairs_have_drawn, ends_dict)
                    cross_pairs_have_drawn[cross_pair] = {'score': cross_continuity,
                                                          'line_color': line_color}
                    if len(intersect_pairs) > 0:
                        for intersect_pair in intersect_pairs:
                            if cross_pairs_have_drawn[intersect_pair]['score'] < cross_continuity: # 原有交叉比新交叉连续性更好,重新绘制一次原有交叉
                                self.drawCross(intersect_pair, ends_dict, routes, cross_pairs_have_drawn[intersect_pair]['line_color'], back_w)

                    if wait_for_redraw:
                        wait_for_redraw = False
                        self.drawRoute(route_pair, ends_dict, routes, route_pairs_have_drawn[route_pair]['line_color'], back_w)
                        for connected_pair in route_pairs_have_drawn[route_pair]['connected_pairs']:
                            self.drawCross(connected_pair, ends_dict, routes, cross_pairs_have_drawn[connected_pair]['line_color'], back_w)

            cv2.imwrite('data/debug_results/route_test/route_comb_visib_{}.jpg'.format(color_index), back_w)
            color_index += 1
        return back_w

    def showCombRoutes_Single(self, skel, routes, ends_dict, combEnds_list):
        route_img_list = []
        for j, singleRoute in enumerate(combEnds_list):
            back = np.zeros_like(skel)
            turn = True
            for k in range(len(singleRoute) - 1):
                point_label = singleRoute[k]
                if turn:
                    turn = False
                    route_label = ends_dict[point_label]['route_label']
                    for point in routes[route_label]['route']:
                        back[point[0]][point[1]] = 255
                else:
                    turn = True
                    next_label = singleRoute[k + 1]
                    end_p1 = ends_dict[point_label]['point']
                    end_p2 = ends_dict[next_label]['point']
                    cv2.line(back, (end_p1[1], end_p1[0]), (end_p2[1], end_p2[0]), color=255, thickness=1)
            route_img_list.append(back)
        return route_img_list

    def calcRouteContinuity(self, routes, singleRoute, k, ends_dict, rgb):
        part_route_score = 0
        if len(singleRoute) >= 4:
            k_left = max(0, k - 2)
            k_right = min(k+3, len(singleRoute)-1)
            partRoute = singleRoute[k_left:k_right+1]
            part_route_score = self.calcRouteScore(routes, partRoute, ends_dict, rgb)
        return part_route_score

    def calcCrossContinuity(self, cross_pair, ends_dict, rgb):
        diff_hsv = 0
        p1 = ends_dict[cross_pair[0]]['point']
        p2 = ends_dict[cross_pair[1]]['point']
        p1_2 = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        IMG_H, IMG_W = rgb.shape[0], rgb.shape[1]
        if self.if_debug:
            top = max(0, min(p1[0]-30, p2[0]-30))
            bottom = min(IMG_H-1, max(p1[0]+30, p2[0]+30))
            left = max(0, min(p1[1]-30, p2[1]-30))
            right = min(IMG_W-1, max(p1[1]+30, p2[1]+30))
            temp_rgb = cv2.cvtColor(rgb, cv2.COLOR_HSV2BGR)
            cv2.circle(temp_rgb, (p1[1], p1[0]), 3, (255, 0, 0), thickness=-1)
            cv2.circle(temp_rgb, (p2[1], p2[0]), 3, (0, 255, 0), thickness=-1)
            cv2.circle(temp_rgb, (p1_2[1], p1_2[0]), 3, (0, 0, 255), thickness=-1)
            cv2.imwrite('data/debug_results/route_test/cross_window.jpg', temp_rgb[top:bottom, left:right])
        hsv_p1 = ends_dict[cross_pair[0]]['end_hsv']
        # hsv_p1_ = rgb[p1[0]][p1[1]]
        hsv_p2 = ends_dict[cross_pair[1]]['end_hsv']
        # hsv_p1_2 = rgb[p1_2[0]][p1_2[1]]
        # diff_hsv += self.costHSV(hsv_p1, hsv_p1_2)
        # diff_hsv += self.costHSV(hsv_p1_2, hsv_p2)
        path_x = list(map(int, list(np.linspace(p1[0], p2[0], num=max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))))))
        path_y = list(map(int, list(np.linspace(p1[1], p2[1], num=max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))))))
        diff_hsv = rgb[path_x, path_y].std(axis=0).sum() / len(path_x)
        return diff_hsv

    def CrossExist(self, cross_pair, cross_pairs_have_drawn, ends_dict):
        intersect_pairs = []
        p1 = ends_dict[cross_pair[0]]['point']
        p2 = ends_dict[cross_pair[1]]['point']
        for cross_pair_drawn in cross_pairs_have_drawn.keys():
            p3 = ends_dict[cross_pair_drawn[0]]['point']
            p4 = ends_dict[cross_pair_drawn[1]]['point']
            if self.EverCross([p1[0], p1[1], p2[0], p2[1]], [p3[0], p3[1], p4[0], p4[1]]):
                intersect_pairs.append(cross_pair_drawn)
        return intersect_pairs


    def EverCross(self, l1, l2):
        v1 = (l1[0] - l2[0], l1[1] - l2[1])
        v2 = (l1[0] - l2[2], l1[1] - l2[3])
        v0 = (l1[0] - l1[2], l1[1] - l1[3])
        a = v0[0] * v1[1] - v0[1] * v1[0]
        b = v0[0] * v2[1] - v0[1] * v2[0]

        temp = l1
        l1 = l2
        l2 = temp
        v1 = (l1[0] - l2[0], l1[1] - l2[1])
        v2 = (l1[0] - l2[2], l1[1] - l2[3])
        v0 = (l1[0] - l1[2], l1[1] - l1[3])
        c = v0[0] * v1[1] - v0[1] * v1[0]
        d = v0[0] * v2[1] - v0[1] * v2[0]

        if a * b < 0 and c * d < 0:
            return True
        else:
            return False


    def drawRoute(self, end_pair, ends_dict, routes, line_color, back_w):
        route_label = ends_dict[end_pair[0]]['route_label']
        radius = routes[route_label]['width'][0]
        for point in routes[route_label]['route']:
            # 非标准绘制模式
            cv2.circle(back_w, (point[1], point[0]), max(round(radius), 1), line_color, -1)
            # 标准绘制模式
            # cv2.circle(back_w, (point[1], point[0]), round(dist_img[point[0]][point[1]]), line_color, -1)
        if self.if_debug:
            cv2.imwrite('data/debug_results/route_test/temp_route.jpg', back_w)


    def drawCross(self, end_pair, ends_dict, routes, line_color, back_w):
        route_label = ends_dict[end_pair[1]]['route_label']
        end_p1 = ends_dict[end_pair[0]]['point']
        end_p2 = ends_dict[end_pair[1]]['point']
        # line_thickness = max((round(routes[route_label]['route_width'])-1) * 2, 3)
        radii_p1 = ends_dict[end_pair[0]]['end_radius']
        radii_p2 = ends_dict[end_pair[1]]['end_radius']
        line_thickness = max(int(radii_p1+radii_p2), 3)
        # 标准绘制模式
        cv2.line(back_w, (end_p1[1], end_p1[0]), (end_p2[1], end_p2[0]), line_color, thickness=line_thickness)

        # # 非标准绘制模式
        # for int_dict in ints_dict.values():
        #     int_ends = int_dict['int_ends']
        #     if end_pair[0] in int_ends:
        #         int_point = int_dict['point']
        #         break
        # cv2.line(back_w, (end_p1[1], end_p1[0]), (int_point[1], int_point[0]), line_color, thickness=line_thickness)
        # cv2.line(back_w, (int_point[1], int_point[0]), (end_p2[1], end_p2[0]), line_color, thickness=line_thickness)
        if self.if_debug:
            cv2.imwrite('data/debug_results/route_test/temp_route.jpg', back_w)

    def combEndsToRoutes(self, rgb, routes, ends_dict, ints_dict, end_pairs, combEnds_list):
        # 单线快速绘制模式
        Route_Drawing_dict = {}
        Cross_Drawing_dict = collections.defaultdict(list)
        back_w = np.zeros_like(rgb)
        color_index = 0
        for j, singleRoute in enumerate(combEnds_list):
            avg_route_r = 0
            avg_route_n = 0
            wait_routes = []
            turn = True
            for k in range(len(singleRoute) - 1):
                if turn:
                    turn = False
                    route_pair_ = [singleRoute[k], singleRoute[k + 1]]
                    route_pair = tuple(sorted(route_pair_))
                    route_label = ends_dict[route_pair[0]]['route_label']
                    Route_Drawing_dict[route_label] = color_index
                    avg_route_r += len(routes[route_label]['route']) * routes[route_label]['width'][0]
                    avg_route_n += len(routes[route_label]['route'])
                    wait_routes.append(route_label)
                else:
                    turn = True
                    cross_pair_ = [singleRoute[k], singleRoute[k + 1]]
                    cross_pair = tuple(sorted(cross_pair_))
                    cross_label = self.GetCrossLabel(cross_pair, end_pairs)
                    cross_continuity = self.calcCrossContinuity(cross_pair, ends_dict, rgb)
                    Cross_Drawing_dict[cross_label].append([cross_pair, color_index, cross_continuity])
            color_index += 1
            for wait_route in wait_routes:
                routes[wait_route]['route_width'] = avg_route_r / avg_route_n + 1

        for route_label in Route_Drawing_dict.keys():
            color_index = Route_Drawing_dict[route_label]
            line_color = (int(self.cmap[color_index][0]),
                          int(self.cmap[color_index][1]),
                          int(self.cmap[color_index][2]))
            self.drawRoute_label(route_label, routes, line_color, back_w)

        for cross_label in Cross_Drawing_dict.keys():
            Cross_Drawing_list = Cross_Drawing_dict[cross_label]
            Cross_Drawing_list.sort(key=lambda x:x[2], reverse=True)
            for cross_info in Cross_Drawing_list:
                color_index = cross_info[1]
                line_color = (int(self.cmap[color_index][0]),
                              int(self.cmap[color_index][1]),
                              int(self.cmap[color_index][2]))
                self.drawCross(cross_info[0], ends_dict, routes, line_color, back_w)

        if self.if_debug:
            cv2.imwrite('data/debug_results/route_test/route_cross_visib_{}.jpg'.format(color_index), back_w)
        return back_w

    def GetCrossLabel(self, cross_pair, end_pairs):
        for end_pair in end_pairs:
            if (cross_pair[0] == end_pair[1] and cross_pair[1] == end_pair[2]) or (cross_pair[0] == end_pair[2] and cross_pair[1] == end_pair[1]):
                cross_label = end_pair[0]
                break
            else:
                cross_label = -1
        return cross_label

    def drawRoute_label(self, route_label, routes, line_color, back_w):
        radius = routes[route_label]['width'][0]
        for point in routes[route_label]['route']:
            cv2.circle(back_w, (point[1], point[0]), max(round(radius), 1), line_color, -1)
        if self.if_debug:
            cv2.imwrite('data/debug_results/route_test/route_cross_visib_temp.jpg', back_w)
            print(1)

    def drawCross_label(self, end_pair, ends_dict, ints_dict, routes, line_color, back_w):
        end_p1 = ends_dict[end_pair[0]]['point']
        end_p2 = ends_dict[end_pair[1]]['point']
        radii_p1 = routes[ends_dict[end_pair[0]]['route_label']]['width'][0]
        radii_p2 = routes[ends_dict[end_pair[1]]['route_label']]['width'][0]

        int_point = ((end_p1[0] + end_p2[0])//2, (end_p1[1] + end_p2[1])//2)
        for int_dict in ints_dict.values():
            int_ends = int_dict['int_ends']
            if end_pair[0] in int_ends:
                int_point = int_dict['point']
                break
        cv2.line(back_w, (end_p1[1], end_p1[0]), (int_point[1], int_point[0]), line_color, thickness=max(round(radii_p1)*2,3))
        cv2.line(back_w, (int_point[1], int_point[0]), (end_p2[1], end_p2[0]), line_color, thickness=max(round(radii_p2)*2,3))
        if self.if_debug:
            cv2.imwrite('data/debug_results/route_test/temp_route.jpg', back_w)
            print(1)