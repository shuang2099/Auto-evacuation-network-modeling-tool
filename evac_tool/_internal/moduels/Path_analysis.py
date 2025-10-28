import math
import os

import cv2
import time
from PIL import Image
import numpy as np
from multiprocessing import Pool
from scipy.spatial import KDTree
import heapq


# 四方向，不允许斜向移动


# move_directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]  # 八方向，允许斜向移动


# 启发式函数
def heuristic(start, end, map_data):
    x1, y1 = start
    x2, y2 = end
    return abs(x1 - x2) + abs(y1 - y2)  # 曼哈顿距离，适用于四方向,无惩罚


# A*算法
def bidirectional_astar(start, end, map_data, move_directions=None):
    """
    双向A*算法实现
    参数:
    start: 起点坐标 (x, y)
    end: 终点坐标 (x, y)
    map_data: 地图数据数组，0表示可通行区域，1表示障碍物
    返回:
    从起点到终点的最短路径，若无路径则返回None
    """
    # 如果起点或终点是障碍物，直接返回None
    if move_directions is None:
        move_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if map_data[start[1], start[0]] != 0 or map_data[end[1], end[0]] != 0:
        return None

    # 使用两个优先队列
    forward_open = []
    backward_open = []
    heapq.heappush(forward_open, (0, start))
    heapq.heappush(backward_open, (0, end))

    # 记录来源节点
    forward_came_from = {}
    backward_came_from = {}

    # 记录实际代价
    forward_g_score = {start: 0}
    backward_g_score = {end: 0}

    # 缓存启发式计算结果
    heuristic_cache = {}

    # 相遇点
    meeting_point = None

    while forward_open and backward_open:
        # 前向搜索一步
        _, current_f = heapq.heappop(forward_open)

        # 检查是否在反向路径中相遇
        if current_f in backward_g_score:
            meeting_point = current_f
            break

        # 检查邻居节点
        for dx, dy in move_directions:
            neighbor = (current_f[0] + dx, current_f[1] + dy)

            # 检查边界和障碍物
            if not (0 <= neighbor[0] < map_data.shape[1] and 0 <= neighbor[1] < map_data.shape[0]):
                continue
            if map_data[neighbor[1], neighbor[0]] != 0:
                continue

            # 计算新的g值
            tentative_g_f = forward_g_score[current_f] + 1

            # 如果找到更好的路径
            if tentative_g_f < forward_g_score.get(neighbor, float('inf')):
                forward_came_from[neighbor] = current_f
                forward_g_score[neighbor] = tentative_g_f

                # 计算或获取启发式值
                if neighbor not in heuristic_cache:
                    heuristic_cache[neighbor] = heuristic(neighbor, end, map_data)

                f_score_f = tentative_g_f + heuristic_cache[neighbor]
                heapq.heappush(forward_open, (f_score_f, neighbor))

        # 后向搜索一步
        _, current_b = heapq.heappop(backward_open)

        # 检查是否在前向路径中相遇
        if current_b in forward_g_score:
            meeting_point = current_b
            break

        # 检查邻居节点
        for dx, dy in move_directions:
            neighbor = (current_b[0] + dx, current_b[1] + dy)

            # 检查边界和障碍物
            if not (0 <= neighbor[0] < map_data.shape[1] and 0 <= neighbor[1] < map_data.shape[0]):
                continue
            if map_data[neighbor[1], neighbor[0]] != 0:
                continue

            # 计算新的g值
            tentative_g_b = backward_g_score[current_b] + 1

            # 如果找到更好的路径
            if tentative_g_b < backward_g_score.get(neighbor, float('inf')):
                backward_came_from[neighbor] = current_b
                backward_g_score[neighbor] = tentative_g_b

                # 计算或获取启发式值
                if neighbor not in heuristic_cache:
                    heuristic_cache[neighbor] = heuristic(neighbor, start, map_data)

                f_score_b = tentative_g_b + heuristic_cache[neighbor]
                heapq.heappush(backward_open, (f_score_b, neighbor))

    # 如果没有相遇点，返回None
    if meeting_point is None:
        return None

    # 重构路径
    # 从前向路径重构从起点到相遇点的路径
    forward_path = []
    current = meeting_point
    while current != start:
        forward_path.append(current)
        current = forward_came_from[current]
    forward_path.append(start)
    forward_path.reverse()

    # 从后向路径重构从相遇点到终点的路径
    backward_path = []
    current = meeting_point
    while current != end:
        current = backward_came_from[current]
        backward_path.append(current)

    # 合并路径
    full_path = forward_path + backward_path
    return full_path


def bidirectional_astar_with_fallback(start, end, map_data, max_nodes=1000000):
    """
    带回退策略的双向A*算法
    参数:
    start, end: 起点和终点坐标
    map_data: 地图数据
    max_nodes: 最大探索节点数
    返回:
    最短路径或单向A*作为回退的路径
    """
    # 尝试双向A*
    path = bidirectional_astar(start, end, map_data)
    if path is not None:
        return path

    # 如果双向A*失败，使用原始A*作为回退
    print("双向A*失败，使用原始A*作为回退")
    return astar(start, end, map_data)


def astar(start, end, map_data):
    move_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    open_list = [(0, start)]  # 优先队列，存放节点和估计代价的元组
    came_from = {}  # 用于记录路径
    g_score = {cell: float('inf') for row in map_data for cell in row}  # 记录从起点到各个节点的实际代价
    g_score[start] = 0

    while open_list:
        _, current = heapq.heappop(open_list)  # 弹出估计代价最小的节点
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in move_directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < map_data.shape[1] and 0 <= neighbor[1] < map_data.shape[0]:
                if map_data[neighbor[1], neighbor[0]] == 0:

                    tentative_g_score = g_score[current] + 1

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        g_score[neighbor] = tentative_g_score
                        came_from[neighbor] = current
                        f_score = g_score[neighbor] + heuristic(neighbor, end, map_data)
                        heapq.heappush(open_list, (f_score, neighbor))
    return None


def astar_plot(path, image, iter_a, out_path):
    # 在原始图像上可视化路径
    map_with_path = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in path:
        cv2.circle(map_with_path, (x, y), 1, (0, 0, 255), -1)  # 在路径点上画红色圆点
    start = path[0]
    end = path[-1]
    # 标记起点和目标点
    cv2.circle(map_with_path, start, 5, (0, 255, 0), -1)  # 起点标记为绿色圆点
    cv2.circle(map_with_path, end, 5, (0, 0, 255), -1)  # 目标点标记为红色圆点
    b = str(next(iter_a))
    cv2.imwrite(out_path['path_img'] + b + ".png", map_with_path)


def path_length(path):
    if len(path) <= 1:
        return 0

    # 预先计算所有连续点之间的向量
    deltas = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]

    # 计算水平/垂直移动数量和对角移动数量
    straight_moves = sum(1 for dx, dy in deltas if dx == 0 or dy == 0)
    diagonal_moves = len(deltas) - straight_moves

    return straight_moves * 1.0 + diagonal_moves * 1.4


def is_exist(arr, element):  # 只判断是否存在
    a = 0
    for i in range(len(arr)):
        if arr[i] == element:
            a = 1
    return a


def find_index(arr, value):  # 返回找到的位置
    for i in range(len(arr)):
        if arr[i] == value:
            return i
    return -1


def count_element(arr, element):
    count = 0
    for row in arr:
        for item in row:
            if item == element:
                count += 1
    return count


def distinct_pixel(arr):
    new_arr = []
    seen = set()
    for item in arr:
        if item not in seen:
            new_arr.append(item)
            seen.add(item)
    return new_arr


def dist(point1, point2):
    length = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    return length


def get_nearest_point(known_point, points_list):
    # 仅当列表很大时才构建KDTree
    if len(points_list) > 100:
        # 构建一次KDTree
        kdtree = KDTree(points_list)
        distance, index = kdtree.query(known_point)
        return points_list[index]
    else:
        # 小列表线性查找
        return min(points_list, key=lambda x: dist(known_point, x))


def calculate_width(point, n, gray_img_path):  # 计算一个方向上长度的函数
    gray_img = Image.open(gray_img_path)
    size = gray_img.size
    ult = max(size[0], size[1])
    x, y = point[0], point[1]
    for i in range(1, ult):  # 假设每个延长点的距离是1
        x = int(point[0] + n[0] * i)  # 计算沿着n方向延长后的点的坐标
        y = int(point[1] + n[1] * i)
        if gray_img.getpixel((x, y)) == 0:
            break
    return dist(point, (x, y))


def draw_paths_on_image(map_image, paths_list, thickness=2):
    if len(map_image.shape) == 2:
        flow_lines_img = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
    else:
        flow_lines_img = map_image.copy()
    n_paths = len(paths_list)
    if n_paths == 0:
        return flow_lines_img
    hues = np.linspace(0, 1, n_paths, endpoint=False)
    colors = []
    for hue in hues:
        hsv_color = np.array([[[hue * 180, 0.8 * 255, 0.95 * 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple([int(c) for c in bgr_color]))
    for idx, t_path in enumerate(paths_list):
        color = colors[idx]
        path_points = []
        for i in range(len(t_path)):
            point = t_path[i]
            if isinstance(point, np.ndarray):
                point = point.tolist()
            path_points.append(tuple(map(int, point)))
        for i in range(len(path_points) - 1):
            cv2.line(flow_lines_img, path_points[i], path_points[i + 1], color, thickness)
    return flow_lines_img


def modified_path_analysis(dirs_path, find_mode='all possible rodes', output_mode=1, middle_thresh=16):
    start_time = time.time()
    """
    执行改进的路径分析算法

    参数:
    dirs_path (str): 多层文件夹的基础路径
    find_mode (str): 路径搜索模式 ('all possible rodes' 或 'shortest roads')
    output_mode (int): 输出模式 (0-完整, 1-简化)
    middle_thresh (int): 中间节点的最小计数阈值
    """
    modes = ['all possible rodes', 'shortest roads']
    print('寻路模式：', find_mode)
    print('输出模式--是否简化：', output_mode)

    # 获取所有在该文件夹内的单层文件夹名称列表
    dirs_names = [f for f in os.listdir(dirs_path)]
    exclude_files = ['Output.xlsx', 'Output.xls', 'result.csv', 'stairs.txt',
                     'min_path.txt', 'process.mp4', 'multi_result.png']

    dirs_names = [f for f in dirs_names if f not in exclude_files]

    for dir_num in range(len(dirs_names)):
        fold_dir = os.path.join(dirs_path, dirs_names[dir_num])
        print(f"处理文件夹: {fold_dir}")

        # 构建输入输出路径字典
        input_path = {
            'origin': os.path.join(fold_dir, '2opening/seg.png'),
            'draw': os.path.join(fold_dir, 'route.png'),
            'img_door_Dilate': os.path.join(fold_dir, '2opening/route_0.png'),
            'xy_doors': os.path.join(fold_dir, '3approxPOLY/xy_doors.txt'),
            'xy_rooms': os.path.join(fold_dir, '3approxPOLY/xy_rooms.txt'),
            'different_pixels': os.path.join(fold_dir, '6connection/contours_1.png'),
            'exit_doors': os.path.join(fold_dir, '6connection/exit_doors.txt'),
            'normal_vector': os.path.join(fold_dir, '5width/normal_vector.txt'),
            'stair_points': os.path.join(dirs_path, 'stairs.txt'),
            'width': os.path.join(fold_dir, '5width/width.txt'),
            'width_change_points': os.path.join(fold_dir, '5width/change_points.txt'),
            'route_3': os.path.join(fold_dir, '2opening/route_3.png'),
            'xy_doors_rects': os.path.join(fold_dir, '4route/xy_doors_rects.txt'),
        }

        # 确保输出目录存在
        output_base = os.path.join(fold_dir, '4route')
        os.makedirs(output_base, exist_ok=True)

        out_path = {
            'path_points': os.path.join(output_base, 'path_points.txt'),
            'path_img': os.path.join(output_base, 'path'),
            'middle_doors': os.path.join(output_base, 'middle_doors.txt'),
            'stair_exit_doors': os.path.join(fold_dir, 'stair_exit_doors.txt'),
            'path_length': os.path.join(output_base, 'path_length.txt'),
            'path_details_1': os.path.join(output_base, 'path_details_1.txt'),
            'exit_door_areas': os.path.join(output_base, 'exit_door_areas.txt'),
            'middle_doors_rects': os.path.join(output_base, 'middle_doors_rects.txt'),
        }

        """读取关键节点坐标"""
        paths_dict = {}
        xy_doors = []
        xy_rooms = []
        exit_doors = []
        width_change_points = []
        f = open(input_path['xy_doors'], "r", encoding='utf-8')
        line = f.readline()  # 读取第一行
        while line:
            txt_data = eval(line)  # 可将字符串变为元组
            xy_doors.append(txt_data)  # 列表增加
            line = f.readline()  # 读取下一行
        f = open(input_path['xy_rooms'], "r", encoding='utf-8')
        line = f.readline()  # 读取第一行
        while line:
            txt_data = eval(line)  # 可将字符串变为元组
            xy_rooms.append(txt_data)  # 列表增加
            line = f.readline()  # 读取下一行
        print('xy doors:', len(xy_doors), '\n', 'xy rooms:', len(xy_rooms))
        f.close()
        f = open(input_path['width_change_points'], "r", encoding='utf-8')
        line = f.readline()  # 读取第一行
        while line:
            txt_data = eval(line)  # 可将字符串变为元组
            width_change_points.append(txt_data)  # 列表增加
            line = f.readline()  # 读取下一行
        print('width_change_points:', width_change_points)
        f.close()

        img_different_pixels = Image.open(input_path['different_pixels'])
        door_pixels = []
        for d in xy_doors:
            door_pixels.append(img_different_pixels.getpixel(d))
        print(door_pixels)

        stair_points = []
        """读取楼梯节点的坐标"""
        with open(input_path['stair_points'], "r") as file:
            lines = file.readlines()
            for line in lines:
                stair_point = line.strip()[1:-1].split('), (')
                stair_point = [tuple(map(int, n.split(','))) for n in stair_point]
                stair_points.append(stair_point)
        print('stair points:', stair_points)

        # 读取宽度
        width = []
        with open(input_path['width'], "r") as file:
            for line in file.readlines():
                width.append(int(line.strip()))

        iter_a = iter(np.arange(1, 5000))

        """初始化地图"""
        # 读取灰度图像
        image = cv2.imread(input_path['origin'], cv2.IMREAD_GRAYSCALE)
        wall = cv2.imread(input_path['route_3'], cv2.IMREAD_GRAYSCALE)
        path_drawing = cv2.imread(input_path['draw'], flags=0)  # 用于画流线的二值化网格地图
        # 设置阈值，将灰度图像二值化，将障碍物标记为1，可通行区域标记为0
        threshold = 128  # 大于阈值为0，小于阈值为1
        binary_map = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY_INV)[1]  # INV取反二值化
        binary_wall = cv2.threshold(wall, threshold, 1, cv2.THRESH_BINARY)[1]

        # 创建一个膨胀核 膨胀障碍物区域
        kernel = np.ones((3, 3), np.uint8)
        dilate_wall = cv2.dilate(binary_wall, kernel, iterations=5)

        # 合并膨胀后的墙体与可通行区域地图
        final_map = cv2.bitwise_or(binary_map, dilate_wall)

        # 将二值化图像转化为地图数据（0表示障碍物，1表示可通行）
        map_array_inv = np.array(final_map)
        out0 = []

        """利用门法向量识别可通行区域判断出口"""
        # 打开原法向量文件，转换成所需法向量格式
        with open(input_path['normal_vector'], 'r') as f:
            contents = f.readlines()
        data = [tuple(map(int, line.strip().replace('[', '').replace(']', '').split())) for line in contents]
        normal_vectors = [(x, y) for x, y in data]
        # print('origin vectors:', normal_vectors)
        """求单位法向量"""
        n_doors_vectors = []
        for t in range(len(xy_doors)):
            n = normal_vectors[t]
            n_length = math.sqrt(n[0] ** 2 + n[1] ** 2)
            n = (n[0] / n_length, n[1] / n_length)
            n_doors_vectors.append(n)  # 单位法向量
        """下面利用d单位法向量结合可通行区域seg图判断出口"""
        for t in range(len(xy_doors)):
            point = xy_doors[t]
            unit_vector = n_doors_vectors[t]

            n1 = (unit_vector[0], unit_vector[1])
            n2 = (-unit_vector[0], -unit_vector[1])
            door_width = width[dir_num]
            scale0 = 0.4 * door_width  # 门法向上的尺度，控制检验点与门的法向距离
            l1 = calculate_width(point, n1, gray_img_path=input_path['origin'])  # 计算法向最大距离
            l2 = calculate_width(point, n2, gray_img_path=input_path['origin'])
            if l1 < scale0 or l2 < scale0:
                exit_doors.append(xy_doors[t])

        # 移除可通行宽度改变处节点，法向量会引起歧义
        for lo in range(len(width_change_points)):
            if is_exist(exit_doors, width_change_points[lo]):
                exit_doors.remove(width_change_points[lo])

        with open(input_path['exit_doors'], 'w') as f:  # 重写出口节点
            for t in range(len(exit_doors)):
                f.write(str(exit_doors[t]) + '\n')
        print('exit doors:', exit_doors)
        # print('出口判断完成')

        # 输出出口范围顶点坐标
        f = open(out_path['exit_door_areas'], "w")
        img_door_Dilate = cv2.imread(input_path['img_door_Dilate'], flags=0)  # route_0.png
        ret, binary = cv2.threshold(img_door_Dilate, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for exit_door in exit_doors:
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
                box = cv2.boxPoints(rect).astype(int)
                XY = tuple(map(round, rect[0]))
                if exit_door[0] == XY[0] and exit_door[1] == XY[1]:
                    f.write(str(box) + '\n')
        f.close()

        """将各层楼梯节点加入，调整首层出口为中间节点"""
        with open(input_path['exit_doors'], 'w') as f:  # 重写出口节点
            for t in range(len(exit_doors)):
                f.write(str(exit_doors[t]) + '\n')

        unique_pixels = []
        paths = []

        if dir_num == 0:
            origins = xy_rooms + stair_points[dir_num]
            exits = exit_doors
        else:
            origins = xy_rooms
            exits = stair_points[dir_num]

        # 输出连贯的具体路径（源节点到出口节点）
        f2 = open(out_path['path_details_1'], 'w')
        # 创建进程池
        po = Pool(processes=os.cpu_count() - 1)
        # 存储异步结果
        results = []
        for node1 in origins:  # 遍历每个房间到各出口的路线，为寻找拥堵节点提供信息
            for node2 in exits:
                try:
                    result = po.apply_async(bidirectional_astar_with_fallback, args=(node1, node2, map_array_inv))
                    results.append((node1, node2, result))
                except Exception as e:
                    print(f"Error processing result for ({node1}, {node2}): {e}")
                    continue

        print('\n', "Multiprocess start")
        po.close()
        po.join()
        print("Multiprocess stop")


        for node1, node2, result in results:
            try:
                path = result.get()  # 获取异步任务的结果
                if path is None:
                    continue
                f2.write(str(path) + '\n')
                pixels = []
                paths_dict[node1, node2] = path
                for point in path:
                    pixels.append(img_different_pixels.getpixel(point))
                unique_pixel = list(set(pixels))
                unique_pixel.remove(255)
                unique_pixels.append(unique_pixel)
            except Exception as e:
                print(f"Error processing result for ({node1}, {node2}): {e}")
        f2.close()

        count_doors = []
        middle_doors_pixels = []
        exit_doors_pixels = []
        for t in range(len(exit_doors)):
            exit_doors_pixels.append(img_different_pixels.getpixel(exit_doors[t]))
        for t in range(len(door_pixels)):  # 计算每个门节点被经过的次数
            num = count_element(unique_pixels, door_pixels[t])
            count_doors.append(num)
        for t in range(len(count_doors)):  # 获取中间拥堵节点
            if count_doors[t] > middle_thresh:  # 想要的次数需要乘以一个房间到n个出口可通行的n
                middle_doors_pixels.append(door_pixels[t])
        for t in range(len(exit_doors_pixels)):  # 从中间节点中移除出口节点避免重复
            if is_exist(middle_doors_pixels, exit_doors_pixels[t]):
                middle_doors_pixels.remove(exit_doors_pixels[t])
        f3 = open(out_path['middle_doors'], 'w')  # 输出拥堵节点
        for t in range(len(middle_doors_pixels)):
            f3.write(str(xy_doors[find_index(door_pixels, middle_doors_pixels[t])]) + '\n')
        f3.close()
        print("中间节点输出完成")

        """输出简化路径集"""
        path_details_list = []
        f1 = open(out_path['path_points'], 'w')
        f2 = open(out_path['path_length'], 'w')
        "两种模式，所有路、最短路"
        if find_mode == 'all possible rodes':
            for node1 in origins:  # 遍历每个房间到各出口的路线，输出路线：[起点 中间节点 出口]
                for node2 in exits:
                    try:
                        path = paths_dict[node1, node2]
                    except Exception as e:
                        print(f"Error processing result for ({node1}, {node2}): {e}")
                        continue
                    path_details_list.append(path)

                    try:
                        pixels = []
                        for point in path:  # 输出路线上经过的每一个像素
                            pixels.append(img_different_pixels.getpixel(point))
                        temp = distinct_pixel(pixels)
                        temp.remove(255)

                        # 记录每个temp值在像素序列中的平均索引位置
                        temp_positions = {}
                        last_position = 0
                        end_position = len(pixels)
                        for pixel_value in temp:
                            # 查找所有该像素值出现的位置
                            positions = [i for i, p in enumerate(pixels) if p == pixel_value]
                            # 计算平均值作为关键节点位置
                            temp_positions[pixel_value] = round(sum(positions) / len(positions)) if positions else 0

                        f1.write(str(node1))  # 输出路线到文件
                        if output_mode:  # 是否简化
                            for t in range(len(temp)):
                                if is_exist(middle_doors_pixels, temp[t]):
                                    f1.write(str(xy_doors[find_index(door_pixels, temp[t])]))  # 按顺序输出拥堵节点坐标
                                    t_seg_length = temp_positions[temp[t]] - last_position
                                    f2.write(str(t_seg_length) + ", ")
                                    last_position = temp_positions[temp[t]]
                        else:
                            for t in range(len(temp)):
                                f1.write(str(xy_doors[find_index(door_pixels, temp[t])]))  # 按顺序输出拥堵节点坐标
                                t_seg_length = temp_positions[temp[t]] - last_position
                                f2.write(str(t_seg_length) + ", ")
                                last_position = temp_positions[temp[t]]
                        f1.write(str(node2) + '\n')
                        t_seg_length = end_position - last_position
                        f2.write(str(t_seg_length) + "\n")

                        # astar_plot(path, path_drawing, iter_a, out_path)
                    except Exception as e:
                        print(f"Error processing result for ({node1}, {node2}): {e}")

        if find_mode == 'shortest roads':
            min_nodes = []
            """"找出最短路"""""
            for node1 in origins:  # 遍历每个房间到各出口的路线，输出路线：[起点 中间节点 出口]
                min_length = 99999999999999
                min_node2 = exits[0]
                for node2 in exits:
                    try:
                        path = paths_dict[node1, node2]
                    except Exception as e:
                        print(f"Error processing result for ({node1}, {node2}): {e}")
                        continue
                    path_details_list.append(path)
                    t_length = path_length(path)
                    f2.write(str(t_length) + '\n')
                    if t_length < min_length:
                        min_length = t_length
                        min_node2 = node2
                min_nodes.append(min_node2)

            """输出最短路"""
            for q in range(len(origins)):  # 遍历最短路径点对
                node1 = origins[q]
                node2 = min_nodes[q]
                try:
                    path = paths_dict[node1, node2]
                except Exception as e:
                    print(f"Error processing result for ({node1}, {node2}): {e}")
                    continue
                try:
                    pixels = []
                    for point in path:  # 输出路线上经过的每一个像素
                        pixels.append(img_different_pixels.getpixel(point))
                    temp = distinct_pixel(pixels)
                    temp.remove(255)

                    # 记录每个temp值在像素序列中的平均索引位置
                    temp_positions = {}
                    last_position = 0
                    end_position = len(pixels)
                    for pixel_value in temp:
                        # 查找所有该像素值出现的位置
                        positions = [i for i, p in enumerate(pixels) if p == pixel_value]
                        # 计算平均值作为关键节点位置
                        temp_positions[pixel_value] = round(sum(positions) / len(positions)) if positions else 0

                    f1.write(str(node1))  # 输出路线到文件
                    if output_mode:  # 是否简化
                        for t in range(len(temp)):
                            if is_exist(middle_doors_pixels, temp[t]):
                                f1.write(str(xy_doors[find_index(door_pixels, temp[t])]))  # 按顺序输出拥堵节点坐标
                                t_seg_length = temp_positions[temp[t]] - last_position
                                f2.write(str(t_seg_length) + ", ")
                                last_position = temp_positions[temp[t]]
                    else:
                        for t in range(len(temp)):
                            f1.write(str(xy_doors[find_index(door_pixels, temp[t])]))  # 按顺序输出拥堵节点坐标标
                            t_seg_length = temp_positions[temp[t]] - last_position
                            f2.write(str(t_seg_length) + ", ")
                            last_position = temp_positions[temp[t]]
                    f1.write(str(node2) + '\n')

                    # astar_plot(path, path_drawing, iter_a, out_path)
                except Exception as e:
                    print(f"Error processing result for ({node1}, {node2}): {e}")
            f1.close()
            f2.close()
        end_time = time.time()  # 记录程序结束时间
        run_time = (end_time - start_time) / 60  # 计算程序运行时间
        print("程序运行时间为：{:.2f}分钟".format(run_time))

        # 绘制路线
        flow_lins = draw_paths_on_image(path_drawing, path_details_list)
        cv2.imwrite(fold_dir + '/4route/flow_lines.png', flow_lins)


# 示例调用
if __name__ == '__main__':
    # 指定分析路径
    analysis_path = r"C:\Users\GuYH\Desktop\example\test_folder"

    # 执行路径分析
    modified_path_analysis(
        dirs_path=analysis_path
    )
