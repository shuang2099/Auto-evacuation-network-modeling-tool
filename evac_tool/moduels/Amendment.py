import itertools

import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from multiprocessing import Pool
import shutil


def dist(box1, box2):
    length = math.sqrt((box2[0] - box1[0]) ** 2 + (box2[1] - box1[1]) ** 2)
    return length


def find_index(arr, value):  # 返回找到的位置
    for i in range(len(arr)):
        if arr[i] == value:
            return i
    return -1


def maximum_internal_rectangle(img_gray):  # 最大内嵌矩形递归函数
    bgr_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    for idx, t_contour in enumerate(contours):
        t_contour = contours[idx].reshape(len(contours[idx]), 2)

        rect = []

        for q in range(len(t_contour)):
            x_1, y_1 = t_contour[q]
            for j in range(len(t_contour)):
                x_2, y_2 = t_contour[j]
                area0 = abs(y_2 - y_1) * abs(x_2 - x_1)
                rect.append(((x_1, y_1), (x_2, y_2), area0))

        all_rect = sorted(rect, key=lambda x: x[2], reverse=True)

        if all_rect:
            best_rect_found = False
            index_rect = 0
            nb_rect = len(all_rect)

            while not best_rect_found and index_rect < nb_rect:

                rect = all_rect[index_rect]
                (x_1, y_1) = rect[0]
                (x_2, y_2) = rect[1]

                valid_rect = True

                x = min(x_1, x_2)
                while x < max(x_1, x_2) + 1 and valid_rect:
                    if any(bgr_img[y_1, x]) == 0 or any(bgr_img[y_2, x]) == 0:
                        valid_rect = False
                    x += 1

                y = min(y_1, y_2)
                while y < max(y_1, y_2) + 1 and valid_rect:
                    if any(bgr_img[y, x_1]) == 0 or any(bgr_img[y, x_2]) == 0:
                        valid_rect = False
                    y += 1

                if valid_rect:
                    best_rect_found = True

                index_rect += 1

            if best_rect_found:
                rects.append([x_1, y_1, x_2, y_2])

            else:
                print("No rectangle fitting into the area")

        else:
            print("No rectangle found")

    return rects


def create_contour_gray_img(gray_image, contour):  # 将轮廓绘制在新的等大空白灰度图上
    height, width = gray_image.shape
    blank_image = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(blank_image, [contour], -1, 38, thickness=cv2.FILLED)
    return blank_image


def change_pixels_in_rect(image, input_rects):  # 递归内嵌矩形时改变已分割矩形的像素
    for input_rect in input_rects:
        x1, y1, x2, y2 = input_rect
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        image[ymin:ymax, xmin:xmax] = 0
    return image


def generate_width_change_rect(p, width, direct):  # 点、宽度、方向
    if direct == 0:
        trbox = [[int(p[0] + width / 2 + 3), int(p[1] + 1)], [int(p[0] + width / 2 + 3), int(p[1] - 1)],
                 [int(p[0] - width / 2 - 3), int(p[1] - 1)],
                 [int(p[0] - width / 2 - 3), int(p[1] + 1)]]
    else:
        trbox = [[int(p[0] + 1), int(p[1] + width / 2 + 3)], [int(p[0] + 1), int(p[1] - width / 2 - 3)],
                 [int(p[0] - 1), int(p[1] - width / 2 - 3)],
                 [int(p[0] - 1), int(p[1] + width / 2 + 3)]]
    return trbox


def return_map_road_mid_point(td, point, gray_map, obstacle_pixel):
    if td == 0:  # 矩形长轴竖向
        n1 = (1, 0)
        n2 = (-1, 0)
        l1 = f_n_o_i_o_d(point, n1, gray_map, obstacle_pixel)
        l2 = f_n_o_i_o_d(point, n2, gray_map, obstacle_pixel)
        if abs(l1 - l2) < 4:
            return point

        t_p = [point[0], point[1]]
        if l1 < l2:  # 初始长度：右边长 向左移
            for move in range(100):
                t_p[0] -= 1
                t_delt_last = abs(l1 - l2)
                l1 = f_n_o_i_o_d(t_p, n1, gray_map, obstacle_pixel)
                l2 = f_n_o_i_o_d(t_p, n2, gray_map, obstacle_pixel)
                t_delt_this = abs(l1 - l2)
                if t_delt_this > t_delt_last:  # 循环结束条件
                    t_p[0] += 1
                    break
        elif l1 > l2:  # 初始长度：左边长 向右移
            for move in range(100):
                t_p[0] += 1
                t_delt_last = abs(l1 - l2)
                l1 = f_n_o_i_o_d(t_p, n1, gray_map, obstacle_pixel)
                l2 = f_n_o_i_o_d(t_p, n2, gray_map, obstacle_pixel)
                t_delt_this = abs(l1 - l2)
                if t_delt_this > t_delt_last:  # 循环结束条件
                    t_p[0] -= 1
                    break
    elif td != 0:  # td!=0 矩形长轴横向
        n1 = (0, 1)
        n2 = (0, -1)
        l1 = f_n_o_i_o_d(point, n1, gray_map, obstacle_pixel)
        l2 = f_n_o_i_o_d(point, n2, gray_map, obstacle_pixel)
        if abs(l1 - l2) < 4:
            return point

        t_p = [point[0], point[1]]
        if l1 < l2:  # 初始长度：下边长 向上移    # 上下移动没有验证过，可能方向反了
            for move in range(100):
                t_p[1] -= 1
                t_delt_last = abs(l1 - l2)
                l1 = f_n_o_i_o_d(t_p, n1, gray_map, obstacle_pixel)
                l2 = f_n_o_i_o_d(t_p, n2, gray_map, obstacle_pixel)
                t_delt_this = abs(l1 - l2)
                if t_delt_this > t_delt_last:  # 循环结束条件
                    t_p[1] += 1
                    break
        elif l1 > l2:  # 初始长度：上边长 向下移
            for move in range(100):
                t_p[1] += 1
                t_delt_last = abs(l1 - l2)
                l1 = f_n_o_i_o_d(t_p, n1, gray_map, obstacle_pixel)
                l2 = f_n_o_i_o_d(t_p, n2, gray_map, obstacle_pixel)
                t_delt_this = abs(l1 - l2)
                if t_delt_this > t_delt_last:  # 循环结束条件
                    t_p[1] -= 1
                    break
    result_point = (t_p[0], t_p[1])
    return result_point


def find_width_change_points_of_complex_areas(check_points_list, length_results_list, thresh, gray_img_path,
                                              obstacle_pixel):  # 查找某块已用最大内嵌矩形分割区域的通行宽度改变处
    gray_img = Image.open(gray_img_path)
    change_mid_points = []
    mid_width = []
    temp_check = []
    for ln in range(len(check_points_list)):
        check_points = check_points_list[ln]
        length_results = length_results_list[ln]
        if ln == 0:
            gradient_width_change_points_indexes = find_gradient_width_change_point(length_results)
            if len(gradient_width_change_points_indexes) > 0:
                gradient_width_change_points = [check_points[num] for num in
                                                find_max_index_of_continuous_integers(
                                                    gradient_width_change_points_indexes)]
                gradient_width_change_points_width = [length_results[num] for num in
                                                      find_max_index_of_continuous_integers(
                                                          gradient_width_change_points_indexes)]
                for ti in range(len(gradient_width_change_points)):
                    if gradient_width_change_points_width[ti] < 120:
                        change_mid_points.append(gradient_width_change_points[ti])
                        mid_width.append(gradient_width_change_points_width[ti])
        for num in range(len(length_results) - 1):
            td = check_points[num + 1][0] - check_points[num][0]  # 用来判断横纵坐标哪个在检查点的一条线上不变
            if abs(length_results[num + 1] - length_results[num]) > thresh and 5 < min(length_results[num],
                                                                                       length_results[num + 1]) < 120:
                if length_results[num + 1] > length_results[num]:
                    change_point = check_points[num]
                elif length_results[num + 1] < length_results[num]:
                    change_point = check_points[num + 1]
                if td == 0:  # 该线上0坐标不变 不同线上同一宽变处1坐标一致 比较不同线之间的1坐标是否一致
                    if change_point[1] in temp_check:
                        continue
                    else:
                        temp_check.append(change_point[1])
                        if ln == 0:
                            result_point = change_point
                        else:
                            # result_point = change_point
                            result_point = return_map_road_mid_point(td, change_point, gray_img, obstacle_pixel)
                        change_mid_points.append(result_point)
                        mid_width.append(min(length_results[num], length_results[num + 1]))

                else:  # 该线上0坐标变化 1坐标不变 不同线上同一宽变处0坐标一致 比较不同线之间的0坐标是否一致
                    if change_point[0] in temp_check:
                        continue
                    else:
                        temp_check.append(change_point[0])
                        if ln == 0:
                            result_point = change_point
                        else:
                            # result_point = change_point
                            result_point = return_map_road_mid_point(td, change_point, gray_img, obstacle_pixel)
                        change_mid_points.append(result_point)
                        mid_width.append(min(length_results[num], length_results[num + 1]))

    return change_mid_points, mid_width


def calculate_rect_2side_width(rect, gray_img_path, obstacle_pixel):  # 计算矩形长轴内可同行宽度
    gray_img = Image.open(gray_img_path)
    x1, y1, x2, y2 = rect
    t_rect_p1 = [x1, y1]
    t_rect_p2 = [x1, y2]
    t_rect_p3 = [x2, y2]
    t_rect_p4 = [x2, y1]
    directions = []  # 0横着1竖着
    if y1 > y2:
        te = y2
        y2 = y1
        y1 = te
    if x1 > x2:
        te = x2
        x2 = x1
        x1 = te

    if dist(t_rect_p1, t_rect_p2) > dist(t_rect_p2, t_rect_p3):
        n = [1, 0]  # 最大内嵌矩形方向与坐标系平行
        check_start_1 = [int((x1 + x2) / 2), y2 + 5]  # 矩形长轴
        check_end_1 = [int((x1 + x2) / 2), y1 - 5]
        check_start_2 = [int((x1 + x2) / 2 + abs(x2 - x1) / 4), y2 + 5]  # 矩形长轴右移1/4宽
        check_end_2 = [int((x1 + x2) / 2 + abs(x2 - x1) / 4), y1 - 5]
        check_start_3 = [int((x1 + x2) / 2 - abs(x2 - x1) / 4), y2 + 5]  # 矩形长轴左移1/4宽
        check_end_3 = [int((x1 + x2) / 2 - abs(x2 - x1) / 4), y1 - 5]

        directions.append(0)
    else:
        n = [0, 1]
        check_start_1 = [x1 - 5, int((y1 + y2) / 2)]
        check_end_1 = [x2 + 5, int((y1 + y2) / 2)]
        check_start_2 = [x1 - 5, int((y1 + y2) / 2 + abs(y2 - y1) / 4)]
        check_end_2 = [x2 + 5, int((y1 + y2) / 2 + abs(y2 - y1) / 4)]
        check_start_3 = [x1 - 5, int((y1 + y2) / 2 - abs(y2 - y1) / 4)]
        check_end_3 = [x2 + 5, int((y1 + y2) / 2 - abs(y2 - y1) / 4)]
        directions.append(1)
    n1 = (n[0], n[1])  # 两个单位向量
    n2 = (-n[0], -n[1])

    # 计算需要计算的点集
    check_points_list = []
    check_points_1 = get_line_coordinates(check_start_1, check_end_1)
    check_points_2 = get_line_coordinates(check_start_2, check_end_2)
    check_points_3 = get_line_coordinates(check_start_3, check_end_3)
    # check_points = find_longest_continuous_path(check_points, gray_img, obstacle_pixel)  # 没发挥作用
    check_points_list.append(check_points_1)
    check_points_list.append(check_points_2)
    check_points_list.append(check_points_3)

    # 计算左右两边可同行宽度并相加
    length_results_list = []
    for check_points in check_points_list:
        t_length_1s = [f_n_o_i_o_d(point, n1, gray_img, obstacle_pixel) for point in check_points]
        t_length_2s = [f_n_o_i_o_d(point, n2, gray_img, obstacle_pixel) for point in check_points]
        len_1s = np.array(t_length_1s)
        len_2s = np.array(t_length_2s)
        length_results = len_1s + len_2s
        length_results_list.append(length_results)

    return check_points_list, length_results_list, directions


def find_gradient_width_change_point(arr):
    special_numbers = []
    for t_i in range(10, len(arr) - 10):
        if (arr[t_i - 10] >= arr[t_i - 9] >= arr[t_i - 8] >= arr[t_i - 7] >= arr[t_i - 6] >= arr[t_i - 5] >= arr[
            t_i - 4] >= arr[t_i - 3] >= arr[t_i - 2] >= arr[t_i - 1] >= arr[t_i] and
                arr[t_i] <= arr[t_i + 1] <= arr[t_i + 2] <= arr[t_i + 3] <= arr[t_i + 4] <= arr[t_i + 5] <= arr[
                    t_i + 6] <= arr[t_i + 7] <= arr[t_i + 8] <= arr[t_i + 9] <= arr[t_i + 10] and
                (len({arr[t_i - 10], arr[t_i - 9], arr[t_i - 8], arr[t_i - 7], arr[t_i - 6], arr[t_i - 5], arr[t_i - 4],
                      arr[t_i - 3], arr[t_i - 2], arr[t_i - 1]}) > 5 or
                 len({arr[t_i + 10], arr[t_i + 9], arr[t_i + 8], arr[t_i + 7], arr[t_i + 6], arr[t_i + 5], arr[t_i + 4],
                      arr[t_i + 3], arr[t_i + 2], arr[t_i + 1]}) > 5)):
            special_numbers.append(t_i)
    return special_numbers


def find_max_index_of_continuous_integers(arr):
    max_indices = []
    start_index = 0
    for t_i in range(1, len(arr)):
        if arr[t_i] != arr[t_i - 1] + 1:
            max_indices.append(max(range(start_index, t_i), key=arr.__getitem__))
            start_index = t_i
    # 处理最后一组连续整数
    max_indices.append(max(range(start_index, len(arr)), key=arr.__getitem__))
    f_result = [arr[max_indices[i]] for i in range(len(max_indices))]
    return f_result


def f_n_o_i_o_d(point, n, gray_img, obstacle_pixel):  # find_nearest_obstacle_in_one_direction
    size = gray_img.size
    ult = max(size[0], size[1])
    x, y = point[0], point[1]
    for t_i in range(1, ult):  # 假设每个延长点的距离是1
        x = int(point[0] + n[0] * t_i)  # 计算沿着n方向延长后的点的坐标
        y = int(point[1] + n[1] * t_i)
        if gray_img.getpixel((x, y)) == obstacle_pixel:  # 本案例中房间类为128 门类38 障碍物0
            break
    return dist(point, (x, y))


def get_line_coordinates(point1, point2):  # 返回两点连线上的整数坐标值
    x1, y1 = point1
    x2, y2 = point2
    coordinates = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if x1 < x2:
        sx = 1
    else:
        sx = -1
    if y1 < y2:
        sy = 1
    else:
        sy = -1
    err = dx - dy
    while True:
        coordinates.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return coordinates


def remove_close_self_points(points_list, width_list, direction_list, threshold, binary_map):
    to_delete = []
    num_points = len(points_list)
    import math
    from collections import deque

    def distance_to_obstacle(point):
        # 获取地图尺寸
        rows = len(binary_map)
        cols = len(binary_map[0]) if rows > 0 else 0
        # 处理空地图情况
        if rows == 0 or cols == 0:
            return float('inf')
        # 起点坐标（四舍五入取整）
        start_x = int(round(point[0]))
        start_y = int(round(point[1]))
        # 检查起点是否超出边界
        if not (0 <= start_x < rows and 0 <= start_y < cols):
            return float('inf')
        # 如果起点就是障碍物，距离为0
        if binary_map[start_x][start_y] == 0:
            return 0
        # BFS初始化
        visited = [[False] * cols for _ in range(rows)]
        queue = deque()
        directions = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
        # 从起点开始搜索
        queue.append((start_x, start_y))
        visited[start_x][start_y] = True
        distance = 0
        while queue:
            distance += 1  # 开始新的一圈
            level_size = len(queue)
            for _ in range(level_size):
                x, y = queue.popleft()
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny]:
                        if binary_map[nx][ny] == 0:
                            return distance
                        visited[nx][ny] = True
                        queue.append((nx, ny))
        # 整个地图无障碍物
        return float('inf')

    for p in range(num_points):
        for j in range(p + 1, num_points):
            point1 = points_list[p]
            point2 = points_list[j]
            distance = dist(point1, point2)

            if distance < threshold:
                to_delete.append(p if width_list[p] < width_list[j] else j)
            elif (point1[0] - point2[0]) * (point1[1] - point2[1]) == 0 and distance < 2 * threshold:
                dist_p = distance_to_obstacle(point1)
                dist_j = distance_to_obstacle(point2)
                to_delete.append(p if dist_p < dist_j else j)


    # Remove points and corresponding widths
    points_list = [point for q_num, point in enumerate(points_list) if q_num not in to_delete]
    width_list = [width for q_num, width in enumerate(width_list) if q_num not in to_delete]
    direction_list = [direction for q_num, direction in enumerate(direction_list) if q_num not in to_delete]

    return points_list, width_list, direction_list


def remove_close_points_with_doors(points_list, doors_list, width_list, direction_list, threshold):
    to_delete = []

    for p in range(len(points_list)):
        for j in range(len(doors_list)):
            point1 = points_list[p]
            point2 = doors_list[j]
            distance = dist(point1, point2)

            if distance < threshold:
                to_delete.append(p)
            # elif (point1[0] - point2[0]) * (point1[1] - point2[1]) == 0 and distance < 2 * threshold:
            #     to_delete.append(p)

    points_list = [point for q_num, point in enumerate(points_list) if q_num not in to_delete]
    width_list = [width for q_num, width in enumerate(width_list) if q_num not in to_delete]
    direction_list = [direction for q_num, direction in enumerate(direction_list) if q_num not in to_delete]

    return points_list, width_list, direction_list


def calculate_squared_distance(point1, point2):
    return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


def get_box_middle_line(box_contour):  # 获取矩形短边的两个中点
    p1 = box_contour[0]
    p2 = box_contour[1]
    p3 = box_contour[2]
    p4 = box_contour[3]
    edge_midpoints = [
        (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)),
        (int((p2[0] + p3[0]) / 2), int((p2[1] + p3[1]) / 2)),
        (int((p3[0] + p4[0]) / 2), int((p3[1] + p4[1]) / 2)),
        (int((p4[0] + p1[0]) / 2), int((p4[1] + p1[1]) / 2))
    ]
    max_distance = 0
    max_points = None
    # 生成所有点对
    point_pairs = itertools.combinations(edge_midpoints, 2)
    # 寻找相距最远的两个点
    for pair in point_pairs:
        dist = abs(pair[1][0] - pair[0][0]) + abs(pair[1][1] - pair[0][1])
        if dist > max_distance:
            max_distance = dist
            max_points = pair
    return max_points


def get_line_points(p1, p2):
    num_points = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) + 1
    x_values = [int(round(p1[0] + (p2[0] - p1[0]) * i / (num_points - 1))) for i in range(num_points)]
    y_values = [int(round(p1[1] + (p2[1] - p1[1]) * i / (num_points - 1))) for i in range(num_points)]
    return list(zip(x_values, y_values))


"""""""""""""""""""""设置参数、阈值"""""""""""""""""""""
# 筛选面积大于阈值的轮廓
threshold_area = 200000
# 设置最大内嵌矩形个数
max_repeats = 20
# 设置最小内嵌矩形面积阈值默认百分比
min_rect_area = 1000
min_have_people_area = 40000
# 距离过近的中间节点删除阈值
min_distance_door_threshold = 30
# 指定要读取图片的文件夹路径
dirs_path = "C:/Users/GuYH/Desktop/test/temp"
# 指定需要手动分割的区块[[一层]、[二层]、[三层]],填rgroute索引
need_divide_indexes = [[45], [20], [20]]
# need_divide_indexes = []
""""""""""""""""""""""""""""""""""""""""""""""""""


def amendment_floor_plans(dirs_path):
    """按多层分开执行单层程序 - 优化版"""
    # 排除特定文件名
    exclude_files = {
        'Output.xlsx', 'Output.xls', 'result.csv',
        'stairs.txt', 'min_path.txt', 'process.mp4',
        'multi_result.png'
    }
    dirs_names = [f for f in os.listdir(dirs_path)
                  if f not in exclude_files and os.path.isdir(os.path.join(dirs_path, f))]

    with Pool(processes=min(4, os.cpu_count())) as pool:
        args_list = [(dirs_path, dir_name, dir_num)
                     for dir_num, dir_name in enumerate(dirs_names)]
        pool.map(process_single_floor, args_list)


def process_single_floor(args):
    """处理单个楼层 - 用于并行执行"""
    dirs_path, dir_name, dir_num = args
    folder = os.path.join(dirs_path, dir_name, '')
    fold_dir = os.path.join(dirs_path, dir_name)

    # 预定义文件路径（使用字典提高可读性）
    input_path = {
        'origin': f'{folder}route.png',
        'seg_doors': f'{folder}1origin/route_0.png',
        'seg_rooms': f'{folder}1origin/route_1.png',
        'c_route_1': f'{folder}2opening/c_route_1.png',
        'stair_points': f'{dirs_path}/stairs.txt',
    }
    out_path = {
        'img_door_Dilate': f'{folder}2opening/route_0.png',
        'img_room_Open': f'{folder}2opening/route_1.png',
        'img_door_mix_room': f'{folder}2opening/seg.png',
        'xy_doors': f'{folder}3approxPOLY/xy_doors.txt',
        'xy_rooms': f'{folder}3approxPOLY/xy_rooms.txt',
        'approx_doors': f'{folder}3approxPOLY/route_0.png',
        'approx_rooms': f'{folder}3approxPOLY/route_1.png',
        'network_graph': f'{folder}rgroute.png',
        'room_areas': f'{folder}3approxPOLY/room_areas.txt',
        'middle_doors': f'{folder}/{dir_name}/4route/middle_doors.txt',
        'c_route_0': f'{folder}2opening/c_route_0.png',
        'c_route_2': f'{folder}2opening/c_route_2.png',
        'route_3': f'{folder}2opening/route_3.png',
        'temp1': f'{folder}2opening/modified_1.png',
        'temp2': f'{folder}2opening/modified_2.png',
        'xy_doors_middle_lines': f'{folder}4route/xy_doors_middle_lines.txt',
        'xy_doors_rects': f'{folder}4route/xy_doors_rects.txt',

    }

    # 利用提取到的房间区域修改补充语义分割图片
    img_door_Dilate = cv2.imread(out_path['img_door_Dilate'], flags=0)  # route_0.png
    img_room_Open = cv2.imread(out_path['img_room_Open'], flags=0)  # route_1.png
    c_route_1 = cv2.imread(input_path['c_route_1'], flags=0)  # c_route_1.png
    c_route_1_PIL = Image.open(input_path['c_route_1']).convert("L")
    #
    # 修改c_route_1并获得c_route_0(门扇形区域)
    # 读取灰度图和彩图
    color_image = Image.open(input_path['origin'])

    # 获取灰度图中所有的0像素点的坐标
    temp_0 = [(x, y) for x in range(c_route_1_PIL.width) for y in range(c_route_1_PIL.height) if
              c_route_1_PIL.getpixel((x, y)) == 0]

    # 获取彩图中像素值等于(0, 0)位置像素值和像素值为(0, 0, 0)的点的坐标
    temp_back = [(x, y) for x in range(color_image.width) for y in range(color_image.height) if
                 color_image.getpixel((x, y)) == (0, 0, 0, 255)
                 or color_image.getpixel((x, y)) == color_image.getpixel((0, 0))]

    # 在temp_0中删除与temp_back相同的坐标
    temp_0 = list(set(temp_0) - set(temp_back))

    # 创建空白的等大灰度图
    t_c_route_0 = Image.new("L", c_route_1_PIL.size)
    # 将所有temp_0中点赋值为127
    for x, y in temp_0:
        t_c_route_0.putpixel((x, y), 127)

    result_cv2 = np.array(t_c_route_0)
    contours, _ = cv2.findContours(result_cv2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 筛选符合扇形条件的轮廓
    filtered_contours = []
    area_threshold = 30  # 面积阈值
    length_threshold = 20  # 点数阈值
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold and len(contour) > length_threshold:  # 如果设置小于点数阈值可以得到门区域，但是也会得到楼梯区域
            filtered_contours.append(contour)
    # 在空白图上绘制筛选后的轮廓并填充127像素
    c_route_0 = np.zeros_like(result_cv2)
    cv2.drawContours(c_route_0, filtered_contours, -1, 128, thickness=cv2.FILLED)
    cv2.imwrite(out_path['c_route_0'], c_route_0)

    # # 整合c_route_1和c_route_0
    c_route_2 = cv2.bitwise_or(c_route_1, c_route_0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    c_route_2 = cv2.morphologyEx(c_route_2, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(out_path['c_route_2'], c_route_2)  # 保存修改后的图像

    # 获取墙体c_route_3
    c_route_3_points = [(x, y) for x in range(color_image.width) for y in range(color_image.height) if
                        color_image.getpixel((x, y)) == (0, 0, 0, 255)]
    route_3 = Image.new("L", c_route_1_PIL.size)
    for x, y in c_route_3_points:
        route_3.putpixel((x, y), 255)
    route_3 = np.array(route_3)
    kernel = np.ones((3, 3), dtype=np.uint8)
    route_3 = cv2.morphologyEx(route_3, cv2.MORPH_OPEN, kernel, iterations=1)
    q_contours, _ = cv2.findContours(route_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    q_filtered_contours = [cnt for cnt in q_contours if cv2.contourArea(cnt) <= 50]
    cv2.drawContours(route_3, q_filtered_contours, -1, 0, thickness=cv2.FILLED)
    cv2.imwrite(out_path['route_3'], route_3)  # 保存修改后的图像

    # # 整合c_route_1、route_3和img_room_open
    img_room_Open = cv2.bitwise_or(c_route_1, img_room_Open)  # 将c_route_1与route_1 进行按位或操作
    q_contours, _ = cv2.findContours(route_3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_room_Open, q_contours, -1, 0, thickness=cv2.FILLED)
    q_contours, _ = cv2.findContours(img_room_Open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_room_Open, q_contours, -1, 128, thickness=cv2.FILLED)
    # 剔除部分c_route_2中扇形
    filtered_contours = []
    points_threshold = 80
    contours, _ = cv2.findContours(c_route_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 10000:
            continue
        # 检查拟合后的边数（顶点数）
        if len(contour) > points_threshold:
            filtered_contours.append(contour)
    cv2.drawContours(img_room_Open, filtered_contours, -1, 0, thickness=cv2.FILLED)
    contours, _ = cv2.findContours(img_room_Open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 1000:
            filtered_contours.append(contour)
    cv2.drawContours(img_room_Open, filtered_contours, -1, 0, thickness=cv2.FILLED)
    cv2.imwrite(out_path['img_room_Open'], img_room_Open)  # 保存修改后的图像

    # 混合开运算后图像
    ret111, binary111 = cv2.threshold(img_door_Dilate, 100, 255, cv2.THRESH_BINARY)
    ret222, binary222 = cv2.threshold(img_room_Open, 100, 255, cv2.THRESH_BINARY)
    img_door_mix_room = cv2.add(binary111, binary222)
    kernel = np.ones((3, 3), dtype=np.uint8)
    img_door_mix_room = cv2.morphologyEx(img_door_mix_room, cv2.MORPH_CLOSE, kernel, iterations=1)
    q_contours, _ = cv2.findContours(img_door_mix_room, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    q_filtered_contours = [cnt for cnt in q_contours if cv2.contourArea(cnt) <= 350]
    cv2.drawContours(img_door_mix_room, q_filtered_contours, -1, 255, thickness=cv2.FILLED)

    # 输出开运算+膨胀后的图像
    cv2.imwrite(out_path['img_door_mix_room'], img_door_mix_room)
    cv2.imwrite(folder + '5width/' + 'seg.png', img_door_mix_room)  # 复制到骨架计算宽度文件夹中，后续不会使用这张图片，此处暂用！！！！！！！！！
    cv2.imwrite(out_path['img_door_Dilate'], img_door_Dilate)

    #  2 最小矩形拟合、Ramer-Douglas-Peucker算法+opencv2多边形拟合
    #  2.1 房间类别多边形拟合

    # 分配小于面积阈值但需要分割的区域的索引
    need_divide_index = need_divide_indexes[dir_num]  # 可以指定多个

    ret, binary = cv2.threshold(img_room_Open, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xy_rooms = []
    xy_rooms_areas = []

    # 筛选特殊处理的轮廓
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > threshold_area]
    other_contours = [contour for contour in contours if cv2.contourArea(contour) <= threshold_area]
    manual_contours = [other_contours[n] for n in need_divide_index]
    for _ in need_divide_index:
        del other_contours[_]

    # 存放分割区域中找到的可通行宽度改变处节点坐标及宽度
    width_change_points = []
    width_change_points_width = []
    width_change_points_direction = []

    #  2.1.1 处理普通房间区域
    for cnt in other_contours:
        cv2.drawContours(img_room_Open, [cnt], -1, (0, 255, 0), 2)  # 画源轮廓
        epsilon = 0.01 * cv2.arcLength(cnt, True)  # epsilon 为近似度参数，该值需要轮廓的周长信息
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        img0 = cv2.drawContours(img_room_Open, [approx], -1, (255, 255, 0), 2)
        # print(approx)
        # 计算轮廓区域的图像矩。 在计算机视觉和图像处理中，图像矩通常用于表征图像中对象的形状。这些力矩捕获了形状的基本统计特性，包括对象的面积，质心（即，对象的中心（x，y）坐标），方向以及其他所需的特性。
        M = cv2.moments(approx)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = int(cv2.contourArea(cnt))
        if area > 7200:  # 去除普通区域中面积太小的
            xy_rooms.append((cX, cY))
            xy_rooms_areas.append(area)
            # 在图像上绘制中心
            cv2.circle(img_room_Open, (cX, cY), 7, (255, 255, 255), -1)
            # cv2.putText(img_room_Open, "center", (cX - 20, cY - 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    #  2.1.2 处理大厅等大面积复杂区域
    for contour in filtered_contours:
        contour_img = create_contour_gray_img(img_room_Open, contour)
        final_rects = []
        for iter_num in range(max_repeats):
            t_rects = maximum_internal_rectangle(contour_img)
            for t_rect in t_rects:
                x1, y1, x2, y2 = t_rect
                t_rect_area = abs(x1 - x2) * abs(y1 - y2)
                if t_rect_area < min_rect_area:  # 进行新生成的最大内嵌矩形面积阈值判断，不需要面积过小的矩形
                    continue
                # cv2.rectangle(origin_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cX = int((x1 + x2) / 2)
                cY = int((y1 + y2) / 2)
                if t_rect_area > min_have_people_area:
                    # print(t_rect_area)
                    xy_rooms.append((cX, cY))
                    xy_rooms_areas.append(t_rect_area)
                    cv2.circle(img_room_Open, (cX, cY), 7, (255, 255, 255), -1)
                # cv2.putText(origin_img, str(t_rect_area), (cX - 20, cY - 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                final_rects.append(t_rect)
            # 修改轮廓图像，以便下一次迭代
            contour_img = change_pixels_in_rect(contour_img, t_rects)

        # 查找可通行宽度改变处
        for one_rect in final_rects:
            t_check_points, t_length_results, direction = calculate_rect_2side_width(one_rect,
                                                                                     out_path['img_room_Open'], 0)
            t_c_points, t_points_width = find_width_change_points_of_complex_areas(t_check_points, t_length_results, 30,
                                                                                   out_path['img_room_Open'], 0)
            if len(t_c_points) != 0:
                width_change_points += t_c_points
                width_change_points_width += t_points_width
                for _ in range(len(t_c_points)):
                    width_change_points_direction += direction

    #  2.1.3 处理手动标定的待分割区域
    for contour in manual_contours:
        contour_img = create_contour_gray_img(img_room_Open, contour)
        final_rects = []
        t_contour_area = int(cv2.contourArea(contour))
        for iter_num in range(max_repeats):
            t_rects = maximum_internal_rectangle(contour_img)
            for t_rect in t_rects:
                x1, y1, x2, y2 = t_rect
                t_rect_area = abs(x1 - x2) * abs(y1 - y2)
                if t_rect_area < t_contour_area * 0.03:  # 进行新生成的最大内嵌矩形面积阈值判断，不需要面积过小的矩形
                    continue
                cX = int((x1 + x2) / 2)
                cY = int((y1 + y2) / 2)
                # cv2.rectangle(origin_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                if t_rect_area > min_have_people_area:
                    xy_rooms.append((cX, cY))
                    xy_rooms_areas.append(t_rect_area)
                    cv2.circle(img_room_Open, (cX, cY), 7, (255, 255, 255), -1)
                # cv2.putText(origin_img, str(t_rect_area), (cX - 20, cY - 20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                final_rects.append(t_rect)
            # 修改轮廓图像，以便下一次迭代
            contour_img = change_pixels_in_rect(contour_img, t_rects)

        # 查找可通行宽度改变处
        for one_rect in final_rects:
            t_check_points, t_length_results, direction = calculate_rect_2side_width(one_rect,
                                                                                     out_path['img_room_Open'], 0)
            t_c_points, t_points_width = find_width_change_points_of_complex_areas(t_check_points, t_length_results, 30,
                                                                                   out_path['img_room_Open'], 0)
            if len(t_c_points) != 0:
                width_change_points += t_c_points
                width_change_points_width += t_points_width
                for _ in range(len(t_c_points)):
                    width_change_points_direction += direction

    stair_points = []  # 删除与楼梯节点相距过近的房间节点区域
    with open(input_path['stair_points'], "r") as file:
        lines = file.readlines()
        for line in lines:
            stair_point = line.strip()[1:-1].split('), (')
            stair_point = [tuple(map(int, n.split(','))) for n in stair_point]
            stair_points.append(stair_point)
    for stair_p in stair_points[dir_num]:  # 去除楼梯节点与房间节点重合的区域
        min_distance_squared = float('inf')  # 初始化最小平方距离为正无穷大
        closest_point = None  # 初始化最近点为None
        for check_xy in xy_rooms:  # 在xy_rooms中找到距离当前stair_p最近的点
            distance_squared = calculate_squared_distance(stair_p, check_xy)
            if distance_squared < min_distance_squared:
                min_distance_squared = distance_squared
                closest_point = check_xy
        if min_distance_squared < 100 ** 2:  # 如果最小平方距离小于阈值的平方，则从xy_rooms中删除这个点
            # print(min_distance_squared)
            xy_rooms.remove(closest_point)
            xy_rooms_areas.remove(xy_rooms_areas[find_index(xy_rooms, closest_point)])

    #  写入源节点面积
    f9 = open(out_path['room_areas'], 'w')
    for q in range(len(xy_rooms_areas)):
        f9.write(str(xy_rooms_areas[q]) + '\n')
    f9.close()

    #  门类别最小矩形拟合
    ret, binary = cv2.threshold(img_door_Dilate, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black = cv2.cvtColor(np.zeros((img_door_Dilate.shape[1], img_door_Dilate.shape[0]), dtype=np.uint8),
                         cv2.COLOR_GRAY2BGR)  # 生成黑色"幕布"
    xy_doors = []
    for cnt in contours:
        cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)  # 源轮廓
        rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        # print(rect[0])  # 中心点坐标
        box = cv2.boxPoints(rect).astype(int)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
        XY = tuple(map(round, rect[0]))  # 元组转为整形
        xy_doors.append(XY)
        cv2.drawContours(img_door_Dilate, [box], 0, 100, 1)  # 画出来
        cv2.circle(img_door_Dilate, XY, 3, 156, 3)

    # 处理过近的重复节点，自身内部以及 与 门节点重合的点
    new_width_change_points, new_width_change_points_width, new_width_change_points_direction = remove_close_self_points(
        width_change_points, width_change_points_width, width_change_points_direction, min_distance_door_threshold, img_door_mix_room)
    new_width_change_points, new_width_change_points_width, new_width_change_points_direction = remove_close_points_with_doors(
        new_width_change_points, xy_doors, new_width_change_points_width, new_width_change_points_direction,
        min_distance_door_threshold)

    width_change_boxes = []
    for lo in range(len(new_width_change_points)):
        t_box = generate_width_change_rect(new_width_change_points[lo], new_width_change_points_width[lo],
                                           new_width_change_points_direction[lo])
        width_change_boxes.append(t_box)

    # 输出查找到的可通行宽度改变处节点及宽度、ops到文件中
    with open(fold_dir + r'\5width\change_points.txt', 'w') as f5:
        for point in new_width_change_points:
            converted_point = (int(point[0]), int(point[1]))
            f5.write(str(converted_point) + '\n')
    # print("可通行宽度改变处已输出到文件")

    f6 = open(fold_dir + r'\5width\change_points_width.txt', 'w')
    f7 = open(fold_dir + '/5width/ops.txt', 'w')
    for t in range(len(new_width_change_points_width)):
        f6.write(str(new_width_change_points_width[t]) + '\n')
        f7.write(str(0.3) + '\n')
    f6.close()
    f7.close()
    # print("宽度已输出到文件")
    # print("ops已输出到文件")

    f = open(out_path['xy_rooms'], 'w')  # 存储坐标到TXT文档
    for t in xy_rooms:
        f.write(str(t) + "\n")
    f.close()

    # cv2.imshow('room', img_room)
    cv2.imwrite(out_path['approx_rooms'], img_room_Open)

    for lo in range(len(new_width_change_points)):
        xy_doors.append(new_width_change_points[lo])

    with open(out_path['xy_doors'], 'w') as f:
        for q in xy_doors:
            # 显式转换为Python int
            converted_point = (int(q[0]), int(q[1]))
            f.write(str(converted_point) + "\n")
    cv2.imwrite(out_path['approx_doors'], img_door_Dilate)

    #  在原图上画点、编号
    img = Image.open(input_path['origin'])
    draw = ImageDraw.Draw(img)
    #  画门
    a = 10  # 三角形边长
    font1 = ImageFont.truetype('SIMLI.TTF', 12)
    font2 = ImageFont.truetype('SIMLI.TTF', 35)
    door_counter = 0  # 门的编号计数器
    for q in range(len(xy_doors)):
        x, y = xy_doors[q]
        # draw.polygon([x - a / 2, y - a * np.sqrt(3) / 6,
        #               x + a / 2, y - a * np.sqrt(3) / 6,
        #               x, y + a / np.sqrt(3)], fill='blue', outline='purple')  # 画门
        x0 = str(x)
        y0 = str(y)
        text_bbox = draw.textbbox((x, y), '(' + x0 + ',' + y0 + ')', font=font1)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        # draw.text((x - text_width / 2, y + a * np.sqrt(3) / 6), '(' + x0 + ',' + y0 + ')', font=font1, fill='red')# 坐标
        draw.text((x - a, y - 2 * a), str(door_counter), font=font2, fill='blue')
        door_counter += 1

    # 画房间
    #  画房间
    b = 10  # 正方形边长
    room_counter = 0  # 房间的编号计数器
    for q in range(len(xy_rooms)):
        x, y = xy_rooms[q]
        # draw.rectangle((x - b / 2, y - b / 2,
        #                 x + b / 2, y + b / 2), fill='blue', outline='purple', width=5)
        x0 = str(x)
        y0 = str(y)
        text_bbox = draw.textbbox((x, y), '(' + x0 + ',' + y0 + ')', font=font1)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        # draw.text((x - text_width / 2, y + b / 2), '(' + x0 + ',' + y0 + ')', font=font1, fill='red')  # 坐标
        draw.text((x - a, y - 2 * a), str(room_counter), font=font2, fill='green')
        room_counter += 1
    img.save(out_path['network_graph'])

    """得到门宽度文件与法向量文件"""
    img = cv2.imread(out_path['img_door_Dilate'])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 生成黑色"幕布"
    black = cv2.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    f1 = open(fold_dir + r'\5width\width_1.txt', 'w')  # 保存距离文本
    f2 = open(fold_dir + r'\5width\width.txt', 'w')  # 保存宽度文本
    f3 = open(fold_dir + r'\5width\normal_vector.txt', 'w')  # 保存法向量文本
    f1.write('(X,Y)' + '\t' + '\t' + '(宽，高)' + '\n')
    for cnt in contours:
        cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)  # 源轮廓
        rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        """下面计算每个门对应的法向量"""
        box = cv2.boxPoints(rect).astype(int) # 通过boxPoints函数返回rect的四个顶点，box是一个存储四个元组的列表
        if dist(box[0], box[1]) > dist(box[1], box[2]):
            n = box[1] - box[2]
        else:
            n = box[0] - box[1]
        f3.write(str(n) + "\n")
        """输出门中心点坐标及宽度"""
        # print(rect[0])  # 中心点坐标
        f1.write(str(rect[0]) + '\t' + str(rect[1]) + '\n')  # 输出坐标及宽高
        e, f = rect[1]
        e, f = int(e), int(f)
        # print(max(e, f))
        f2.write(str(max(e, f)) + '\n')
        XY = tuple(map(round, rect[0]))  # 元组转为整形
        # 画出来
        cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
        cv2.circle(img, XY, 3, (255, 0, 0), 3)
        # cv2.imwrite(r'C:\Users\13812502988\Desktop\test\3approxPOLY\126_0.png', img)
    f1.close()

    for t in range(len(new_width_change_points_width)):
        f2.write(str(int(new_width_change_points_width[t])) + '\n')
    f2.close()

    # 添加可通行宽度改变处法向量文本
    for lo in range(len(width_change_points_direction)):
        n = [1 - width_change_points_direction[lo], width_change_points_direction[lo]]
        n = np.array(n)
        f3.write(str(n) + "\n")
    f3.close

    """语义分割转换得到实例分割"""
    img = cv2.imread(out_path['img_door_Dilate'])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_contours = []
    # 生成黑色"幕布"
    black = cv2.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)  # 源轮廓
        rect = list(cv2.minAreaRect(cnt))  # 得到”识别的门“最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        if rect[1][0] < rect[1][1]:
            rect[1] = (2, rect[1][1] + 2)  # 修改宽度
        else:
            rect[1] = (rect[1][0] + 2, 2)  # 修改宽度
        box = cv2.boxPoints(tuple(rect)).astype(int)  # 获取最小外接矩形的4个顶点坐标
        box_points = [list(box[0]), list(box[1]), list(box[2]), list(box[3])]
        box_contours.append(box_points)  # 将最小矩形添加到轮廓中

        XY = tuple(map(round, rect[0]))  # 元组转为整形
        # 画出来
        cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
        cv2.circle(img, XY, 3, (255, 0, 0), 3)


    # 给不同门的轮廓赋予不同颜色的像素(三通道)
    contour_img = cv2.imread(fold_dir + "/2opening/seg.png")
    seg_gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY) if contour_img.ndim == 3 else contour_img.copy()

    for lo in range(len(width_change_boxes)):
        box_points = [list(width_change_boxes[lo][0]), list(width_change_boxes[lo][1]), list(width_change_boxes[lo][2]),
                      list(width_change_boxes[lo][3])]
        box_contours.append(box_points)
    box_contours = np.array(box_contours)

    f4 = open(out_path['xy_doors_middle_lines'], 'w')
    f5 = open(out_path['xy_doors_rects'], 'w')
    for q, contour in enumerate(box_contours):
        # 直接在灰度图上操作，直接赋灰度值
        gray_val = (q % 254) + 1  # 灰度值范围1-254
        cnt = contour.reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(seg_gray, [cnt], -1, int(gray_val), thickness=cv2.FILLED)

        t_p_1, t_p_2 = get_box_middle_line(contour)
        line_points = get_line_points(t_p_1, t_p_2)
        f4.write(str(line_points) + '\n')
        f5.write(str(contour) + '\n')
    f4.close()
    f5.close()
    cv2.imwrite(fold_dir + '/6connection/contours_1.png', seg_gray)

# 示例调用
if __name__ == '__main__':
    # 指定分析路径
    analysis_path = "C:/Users/GuYH/Desktop/test/temp"

    # 执行路径分析
    amendment_floor_plans(
        dirs_path=analysis_path
    )
