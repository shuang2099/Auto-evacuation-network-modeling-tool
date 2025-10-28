import time
import math
import numpy as np
import pandas as pd
from numpy import inf
import warnings

warnings.filterwarnings("ignore")


def run_macro_evac(dirs_path):
    # Floyd 算法实现
    def Floyd(a):
        b = a.copy()
        n = b.shape[0]
        path = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if b[i, j] != float(inf):
                    path[i, j] = j
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if b[i, j] > (b[i, k] + b[k, j]):
                        b[i, j] = b[i, k] + b[k, j]
                        path[i, j] = path[i, k]
        dist = b
        return dist, path

    # 速度折减函数
    def ReductionV(N, W, ops):
        if ops == 0.3:
            Rv = np.round(0.066 + 0.473 / N + 0.001 * W - (1.086 / (N ** 2)) - 0.001 * (W ** 2) - 0.178 * (W / N), 3)
        elif ops == 0.4:
            Rv = np.round((61.280 - 17.414 / N + 50.723 * W * math.log(W) - 40.318 * W ** 1.5) ** -1, 3)
        else:
            Rv = 1
            print('不存在折减')
        return Rv

    # 求和函数
    def sump(P):
        return np.sum(P)

    # 找到路径中最后到达节点的函数
    def id_at_last(tai, mid_id, pathlist):
        at_last = tai[pathlist[0][0]][mid_id]
        id_last = pathlist[0][0]
        for j in range(np.size(pathlist)):
            if tai[pathlist[0][j]][mid_id] >= at_last:
                at_last = tai[pathlist[0][j]][mid_id]
                id_last = pathlist[0][j]
        return at_last, id_last

    # 行人移动模拟主函数
    def evatimesp(PLgmin):
        P_Locs = []
        t = 0
        n = 0
        nonlocal node_sum, sourcenum, v0, v, vd

        Cv = np.ones((node_sum, node_sum)) * Cvt
        Rcv = np.ones((node_sum, node_sum))
        nonlocal r_rv

        Nrpn = np.zeros((node_sum, node_sum))
        Nps_wpn = np.zeros((node_sum, node_sum))
        tmw = np.zeros((node_sum, node_sum))
        tai = np.zeros((node_sum, node_sum))

        # 初始化行人位置
        t0_P_Locs = []
        for i in range(sourcenum):
            t0_P_Locs.append(100000 * i)
        P_Locs.append(t0_P_Locs)
        aP_Locs = np.array(P_Locs)

        # 行人路径更新函数
        def perlodem(num_PN, Cur_nodepath, n_rec, tfro, n, t, i):
            # 简化处理 2-4 个节点的路径
            if num_PN == 2:
                tr_e1 = round(TraLen[Cur_nodepath[0], Cur_nodepath[1]] / v, 1)
                ID_e1 = RoaInd.tolist().index([Cur_nodepath[0], Cur_nodepath[1]])
                if t <= tr_e1:
                    return ID_e1  # 仍在第一段路径
                return -1  # 已完成路径

            elif num_PN == 3:
                tr_e1 = round(TraLen[Cur_nodepath[0], Cur_nodepath[1]] / v, 1)
                tr_e2 = round(TraLen[Cur_nodepath[1], Cur_nodepath[2]] / v, 1)
                ID_e1 = RoaInd.tolist().index([Cur_nodepath[0], Cur_nodepath[1]])
                ID_e2 = RoaInd.tolist().index([Cur_nodepath[1], Cur_nodepath[2]])

                # 处理不同时间点的情况
                if t < tfro + tr_e1:
                    return ID_e1
                elif t == tfro + tr_e1:
                    # 到达第一个节点
                    Napn[Cur_nodepath[1]] += per_num[i]
                    tai[i][Cur_nodepath[1]] = t

                    # 处理节点拥塞
                    pathlist = np.where(aP_Locs[n - 1] == 100000 * Cur_nodepath[1])
                    if np.size(pathlist) == 0:
                        Nrpn[i][Cur_nodepath[1]] = Napn[Cur_nodepath[1]]
                    else:
                        at_last, k = id_at_last(tai, Cur_nodepath[1], pathlist)
                        Nps_wpn[i][Cur_nodepath[1]] = Nrpn[k][Cur_nodepath[1]] - (t - tai[k][Cur_nodepath[1]]) * Cv[k][
                            Cur_nodepath[1]]
                        Nrpn[i][Cur_nodepath[1]] = Nps_wpn[i][Cur_nodepath[1]] + Napn[Cur_nodepath[1]]

                    # 检查是否需要等待
                    if Nrpn[i][Cur_nodepath[1]] <= C[Cur_nodepath[1]]:
                        tmw[i][Cur_nodepath[1]] = 0
                        return ID_e2
                    else:
                        Rcv[i][Cur_nodepath[1]] = r_rv * ReductionV(Nrpn[i][Cur_nodepath[1]],
                                                                    all_width[Cur_nodepath[1]],
                                                                    all_ops[Cur_nodepath[1]])
                        Cv[i][Cur_nodepath[1]] = np.round(Rcv[i][Cur_nodepath[1]] * Cvt[Cur_nodepath[1]], 3)
                        tmw[i][Cur_nodepath[1]] = np.round(Nrpn[i][Cur_nodepath[1]] / Cv[i][Cur_nodepath[1]], 1)
                        return 100000 * Cur_nodepath[1]  # 等待状态

                elif t > tfro + tr_e1:
                    if tmw[i][Cur_nodepath[1]] == 0:
                        if t <= tfro + tr_e1 + tr_e2:
                            return ID_e2
                        return -1
                    else:
                        if t <= tfro + tr_e1 + tmw[i][Cur_nodepath[1]]:
                            return 100000 * Cur_nodepath[1]
                        elif t <= tfro + tr_e1 + tmw[i][Cur_nodepath[1]] + tr_e2:
                            return ID_e2
                        return -1

            # 处理 4 个节点的路径
            elif num_PN == 4:
                # 获取路径信息
                ID_e1 = RoaInd.tolist().index([Cur_nodepath[0], Cur_nodepath[1]])
                ID_e2 = RoaInd.tolist().index([Cur_nodepath[1], Cur_nodepath[2]])
                ID_e3 = RoaInd.tolist().index([Cur_nodepath[2], Cur_nodepath[3]])

                # 计算路径时间 (考虑坡道因素)
                tr_e1 = round(TraLen[Cur_nodepath[0], Cur_nodepath[1]] / (vd if PwT[ID_e1] == 1 else v), 1)
                tr_e2 = round(TraLen[Cur_nodepath[1], Cur_nodepath[2]] / (vd if PwT[ID_e2] == 1 else v), 1)
                tr_e3 = round(TraLen[Cur_nodepath[2], Cur_nodepath[3]] / (vd if PwT[ID_e3] == 1 else v), 1)

                # 处理时间点
                if t < tfro + tr_e1:
                    return ID_e1
                elif t == tfro + tr_e1:
                    # 到达第一个节点
                    Napn[Cur_nodepath[1]] += per_num[i]
                    tai[i][Cur_nodepath[1]] = t

                    # 处理节点拥塞
                    pathlist = np.where(aP_Locs[n - 1] == 100000 * Cur_nodepath[1])
                    if np.size(pathlist) == 0:
                        Nrpn[i][Cur_nodepath[1]] = Napn[Cur_nodepath[1]]
                    else:
                        at_last, k = id_at_last(tai, Cur_nodepath[1], pathlist)
                        Nps_wpn[i][Cur_nodepath[1]] = Nrpn[k][Cur_nodepath[1]] - (t - tai[k][Cur_nodepath[1]]) * Cv[k][
                            Cur_nodepath[1]]
                        Nrpn[i][Cur_nodepath[1]] = Nps_wpn[i][Cur_nodepath[1]] + Napn[Cur_nodepath[1]]

                    # 检查是否需要等待
                    if Nrpn[i][Cur_nodepath[1]] <= C[Cur_nodepath[1]]:
                        tmw[i][Cur_nodepath[1]] = 0
                        return ID_e2
                    else:
                        Rcv[i][Cur_nodepath[1]] = r_rv * ReductionV(Nrpn[i][Cur_nodepath[1]],
                                                                    all_width[Cur_nodepath[1]],
                                                                    all_ops[Cur_nodepath[1]])
                        Cv[i][Cur_nodepath[1]] = np.round(Rcv[i][Cur_nodepath[1]] * Cvt[Cur_nodepath[1]], 3)
                        tmw[i][Cur_nodepath[1]] = np.round(Nrpn[i][Cur_nodepath[1]] / Cv[i][Cur_nodepath[1]], 1)
                        return 100000 * Cur_nodepath[1]

                elif t > tfro + tr_e1:
                    if tmw[i][Cur_nodepath[1]] == 0:
                        # 移动状态
                        if t < tfro + tr_e1 + tr_e2:
                            return ID_e2
                        elif t == tfro + tr_e1 + tr_e2:
                            # 到达第二个节点
                            Napn[Cur_nodepath[2]] += per_num[i]
                            tai[i][Cur_nodepath[2]] = t

                            # 处理节点拥塞
                            pathlist = np.where(aP_Locs[n - 1] == 100000 * Cur_nodepath[2])
                            if np.size(pathlist) == 0:
                                Nrpn[i][Cur_nodepath[2]] = Napn[Cur_nodepath[2]]
                            else:
                                at_last, k = id_at_last(tai, Cur_nodepath[2], pathlist)
                                Nps_wpn[i][Cur_nodepath[2]] = Nrpn[k][Cur_nodepath[2]] - (t - tai[k][Cur_nodepath[2]]) * \
                                                              Cv[k][Cur_nodepath[2]]
                                Nrpn[i][Cur_nodepath[2]] = Nps_wpn[i][Cur_nodepath[2]] + Napn[Cur_nodepath[2]]

                            # 检查是否需要等待
                            if Nrpn[i][Cur_nodepath[2]] <= C[Cur_nodepath[2]]:
                                tmw[i][Cur_nodepath[2]] = 0
                                return ID_e3
                            else:
                                Rcv[i][Cur_nodepath[2]] = r_rv * ReductionV(Nrpn[i][Cur_nodepath[2]],
                                                                            all_width[Cur_nodepath[2]],
                                                                            all_ops[Cur_nodepath[2]])
                                Cv[i][Cur_nodepath[2]] = np.round(Rcv[i][Cur_nodepath[2]] * Cvt[Cur_nodepath[2]], 3)
                                tmw[i][Cur_nodepath[2]] = np.round(Nrpn[i][Cur_nodepath[2]] / Cv[i][Cur_nodepath[2]], 1)
                                return 100000 * Cur_nodepath[2]
                        else:  # t > tfro + tr_e1 + tr_e2
                            if tmw[i][Cur_nodepath[2]] == 0:
                                if t <= tfro + tr_e1 + tr_e2 + tr_e3:
                                    return ID_e3
                                return -1
                            else:
                                if t <= tfro + tr_e1 + tr_e2 + tmw[i][Cur_nodepath[2]]:
                                    return 100000 * Cur_nodepath[2]
                                elif t <= tfro + tr_e1 + tr_e2 + tmw[i][Cur_nodepath[2]] + tr_e3:
                                    return ID_e3
                                return -1
                    else:  # 有等待时间
                        if t <= tfro + tr_e1 + tmw[i][Cur_nodepath[1]]:
                            return 100000 * Cur_nodepath[1]
                        else:
                            if t < tfro + tr_e1 + tmw[i][Cur_nodepath[1]] + tr_e2:
                                return ID_e2
                            elif t == tfro + tr_e1 + tmw[i][Cur_nodepath[1]] + tr_e2:
                                # 到达第二个节点
                                Napn[Cur_nodepath[2]] += per_num[i]
                                tai[i][Cur_nodepath[2]] = t

                                # 处理节点拥塞
                                pathlist = np.where(aP_Locs[n - 1] == 100000 * Cur_nodepath[2])
                                if np.size(pathlist) == 0:
                                    Nrpn[i][Cur_nodepath[2]] = Napn[Cur_nodepath[2]]
                                else:
                                    at_last, k = id_at_last(tai, Cur_nodepath[2], pathlist)
                                    Nps_wpn[i][Cur_nodepath[2]] = Nrpn[k][Cur_nodepath[2]] - (
                                                t - tai[k][Cur_nodepath[2]]) * Cv[k][Cur_nodepath[2]]
                                    Nrpn[i][Cur_nodepath[2]] = Nps_wpn[i][Cur_nodepath[2]] + Napn[Cur_nodepath[2]]

                                # 检查是否需要等待
                                if Nrpn[i][Cur_nodepath[2]] <= C[Cur_nodepath[2]]:
                                    tmw[i][Cur_nodepath[2]] = 0
                                    return ID_e3
                                else:
                                    Rcv[i][Cur_nodepath[2]] = r_rv * ReductionV(Nrpn[i][Cur_nodepath[2]],
                                                                                all_width[Cur_nodepath[2]],
                                                                                all_ops[Cur_nodepath[2]])
                                    Cv[i][Cur_nodepath[2]] = np.round(Rcv[i][Cur_nodepath[2]] * Cvt[Cur_nodepath[2]], 3)
                                    tmw[i][Cur_nodepath[2]] = np.round(
                                        Nrpn[i][Cur_nodepath[2]] / Cv[i][Cur_nodepath[2]], 1)
                                    return 100000 * Cur_nodepath[2]
                            else:
                                if tmw[i][Cur_nodepath[2]] == 0:
                                    if t <= tfro + tr_e1 + tmw[i][Cur_nodepath[1]] + tr_e2 + tr_e3:
                                        return ID_e3
                                    return -1
                                else:
                                    if t <= tfro + tr_e1 + tmw[i][Cur_nodepath[1]] + tr_e2 + tmw[i][Cur_nodepath[2]]:
                                        return 100000 * Cur_nodepath[2]
                                    elif t <= tfro + tr_e1 + tmw[i][Cur_nodepath[1]] + tr_e2 + tmw[i][
                                        Cur_nodepath[2]] + tr_e3:
                                        return ID_e3
                                    return -1

            # 通用路径处理 (多于 4 个节点)
            else:
                # 获取第一段路径信息
                ID_e01 = RoaInd.tolist().index([Cur_nodepath[0], Cur_nodepath[1]])
                tr_e01 = round(TraLen[Cur_nodepath[0], Cur_nodepath[1]] / (vd if PwT[ID_e01] == 1 else v), 1)

                # 处理第一段路径
                if t < tfro + tr_e01:
                    return ID_e01
                elif t == tfro + tr_e01:
                    # 到达第一个节点
                    Napn[Cur_nodepath[1]] += per_num[i]
                    tai[i][Cur_nodepath[1]] = t

                    # 处理节点拥塞
                    pathlist = np.where(aP_Locs[n - 1] == 100000 * Cur_nodepath[1])
                    if np.size(pathlist) == 0:
                        Nrpn[i][Cur_nodepath[1]] = Napn[Cur_nodepath[1]]
                    else:
                        at_last, k = id_at_last(tai, Cur_nodepath[1], pathlist)
                        Nps_wpn[i][Cur_nodepath[1]] = Nrpn[k][Cur_nodepath[1]] - (t - tai[k][Cur_nodepath[1]]) * Cv[k][
                            Cur_nodepath[1]]
                        Nrpn[i][Cur_nodepath[1]] = Nps_wpn[i][Cur_nodepath[1]] + Napn[Cur_nodepath[1]]

                    # 检查是否需要等待
                    if Nrpn[i][Cur_nodepath[1]] <= C[Cur_nodepath[1]]:
                        tmw[i][Cur_nodepath[1]] = 0
                        return RoaInd.tolist().index([Cur_nodepath[1], Cur_nodepath[2]])
                    else:
                        Rcv[i][Cur_nodepath[1]] = r_rv * ReductionV(Nrpn[i][Cur_nodepath[1]],
                                                                    all_width[Cur_nodepath[1]],
                                                                    all_ops[Cur_nodepath[1]])
                        Cv[i][Cur_nodepath[1]] = np.round(Rcv[i][Cur_nodepath[1]] * Cvt[Cur_nodepath[1]], 3)
                        tmw[i][Cur_nodepath[1]] = np.round(Nrpn[i][Cur_nodepath[1]] / Cv[i][Cur_nodepath[1]], 1)
                        return 100000 * Cur_nodepath[1]

                elif t > tfro + tr_e01:
                    if tmw[i][Cur_nodepath[1]] == 0:
                        # 移动到下一段路径
                        tfro += tr_e01
                        rec_Cur_nodepath = Cur_nodepath[1:]
                        return perlodem(len(rec_Cur_nodepath), rec_Cur_nodepath, n_rec + 1, tfro, n, t, i)
                    else:
                        # 处理等待时间
                        if t <= tfro + tr_e01 + tmw[i][Cur_nodepath[1]]:
                            return 100000 * Cur_nodepath[1]
                        else:
                            # 等待结束后移动
                            tfro += tr_e01 + tmw[i][Cur_nodepath[1]]
                            rec_Cur_nodepath = Cur_nodepath[1:]
                            return perlodem(len(rec_Cur_nodepath), rec_Cur_nodepath, n_rec + 1, tfro, n, t, i)

        # 主循环 - 模拟行人移动
        while sump(aP_Locs[n]) != -sourcenum:
            t = round(t + 0.1, 1)
            n += 1
            temp_P_Locs = []
            Napn = np.zeros(node_sum)  # 当前节点累积人数

            # 处理每个源节点
            for i in range(sourcenum):
                Cur_nodepath = PLgmin[0][i]
                num_PN = len(Cur_nodepath)
                label_P_Locs_r = perlodem(num_PN, Cur_nodepath, 0, 0, n, t, i)
                temp_P_Locs.append(label_P_Locs_r)

            # 更新位置
            P_Locs.append(temp_P_Locs)
            aP_Locs = np.array(P_Locs)

        # 找到最后一批行人的位置
        t_global = t
        P_ps = np.array(aP_Locs[n - 1])
        kgt = np.where(P_ps != -1)
        kg = kgt[0][0] if kgt[0].size > 0 else -1

        return t_global, kg, aP_Locs, PLgmin, DLgmin

    # ======= 主流程开始 =======
    start = time.time()

    # 从 Excel 读取数据
    file_path = dirs_path + '/Output.xlsx'

    # 源节点数据
    sheetname02 = 'Number of Source Nodes'
    df_num = pd.read_excel(file_path, sheetname02, engine='openpyxl')
    num_ori = df_num.values
    sour_set = num_ori[:, 0]
    sourcenum = sour_set.shape[0]
    source_width = np.ones(sourcenum) * inf
    source_ops = np.ones(sourcenum)
    per_num = num_ori[:, 1]

    # 中间节点数据
    sheetname03 = 'Middle Node capacity'
    df_mid_width = pd.read_excel(file_path, sheetname03, engine='openpyxl')
    mid_width_ori = df_mid_width.values
    mid_width = mid_width_ori[:, 1]
    mid_ops = mid_width_ori[:, 2]
    mid_set = mid_width_ori[:, 0]

    # 出口节点数据
    sheetname04 = 'Exit capacity'
    df_exit_width = pd.read_excel(file_path, sheetname04, engine='openpyxl')
    exit_width_ori = df_exit_width.values
    exit_width = exit_width_ori[:, 1]
    exit_ops = exit_width_ori[:, 2]
    exit_set = exit_width_ori[:, 0]

    # 合并所有节点
    all_node_set = np.concatenate((sour_set, mid_set, exit_set), axis=0)
    node_sum = all_node_set.shape[0]
    all_width = np.concatenate((source_width, mid_width, exit_width), axis=0)
    all_ops = np.concatenate((source_ops, mid_ops, exit_ops), axis=0)

    Asnode = node_sum
    all_n_num = Asnode + 1
    num2 = node_sum + 1

    # 读取路径信息
    sheetname01 = 'Link Information'
    dfa = pd.read_excel(file_path, sheetname01, engine='openpyxl')
    num1 = dfa.values
    Nind1 = num1.shape[0]
    RoaInd = num1[:, 1:3]
    RoaLen = num1[:, 3]
    PwT = num1[:, 4]  # 坡度标识

    # 创建路径长度矩阵
    TraLen = np.ones((all_n_num, all_n_num)) * np.inf
    for i in range(all_n_num):
        for j in range(all_n_num):
            if i == j:
                TraLen[i, j] = 0

    # 填充已知路径长度
    for i in range(Nind1):
        start_node = int(RoaInd[i, 0])
        end_node = int(RoaInd[i, 1])
        TraLen[start_node, end_node] = RoaLen[i]

    # 设置行人参数
    per_size = [0.4, 0.4]
    Dpf = per_size[0]  # 行人直径 (单位:米)
    Dvf = per_size[1]  # 行人宽度
    v0 = 3.124  # 自由流速度 (单位:米/秒)
    v = 0.3 * v0  # 标准流速
    vd = 0.3  # 下坡流速

    # 计算所有节点间最短路径
    Dis, path = Floyd(TraLen)

    # 计算到出口的最短路径
    PLgmin = np.empty((1, sourcenum), dtype=object)
    db = Asnode
    DLgmin = np.ones(sourcenum) * np.inf
    for i in range(sourcenum):
        DLgmin[i] = Dis[i][db] - 1
        path_temp = [i]
        tl = i
        while tl != db:
            temp = int(path[tl, db])
            path_temp.append(temp)
            tl = temp
        PLgmin[0][i] = path_temp[0:-1]  # 排除出口节点

    # 计算备选路径
    exitnum = exit_set.shape[0]
    altpaset = np.empty((sourcenum, exitnum), dtype=object)
    Daltpa = np.ones((sourcenum, exitnum)) * np.inf
    for i in range(sourcenum):
        for j in range(exitnum):
            exit_node = int(exit_set[j])
            Daltpa[i, j] = Dis[i][exit_node]
            if Daltpa[i, j] == inf:
                altpaset[i, j] = np.array([])
            else:
                altpath_temp = [i]
                tt = i
                while tt != exit_node:
                    temp = int(path[tt, exit_node])
                    altpath_temp.append(temp)
                    tt = temp
                altpaset[i, j] = altpath_temp

    # 计算节点容量和服务能力
    C = np.floor(all_width / Dvf)
    Cvt = np.round((all_width / Dvf) / (Dpf / v0), 3)  # 理论服务能力
    r_rv = 1.0  # 折减系数

    # 运行行人移动模型
    t_global0, kg0, aP_Locs0, PLgmin0, DLgmin0 = evatimesp(PLgmin)

    end = time.time()
    run_time = end - start

    # 保存结果
    np.savetxt(dirs_path + '/result.csv', aP_Locs0, delimiter=',')

    with open(dirs_path + '/min_path.txt', 'w') as f:
        for i in range(len(PLgmin[0])):
            f.write(str(PLgmin[0][i]) + '\n')

    print(f"计算完成! 总耗时: {run_time:.2f}秒")
    print(f"最后一批行人移动总时间: {t_global0}秒")

# 调用示例
# dirs_path = "C:/Users/GuYH/Desktop/test/temp"
# run_macro_evac(dirs_path)