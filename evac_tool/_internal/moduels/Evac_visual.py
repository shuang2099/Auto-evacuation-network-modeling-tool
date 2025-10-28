import io
import tempfile

import ffmpeg
import imageio
import matplotlib
from PIL import Image

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import math
import os
import cv2
import matplotlib.animation as animation
import numpy as np
import random
import pandas as pd
from scipy.optimize import linear_sum_assignment


class MultiFloorEvacuationSimulator:
    def __init__(self, dirs_path):
        self.dirs_path = dirs_path
        self.nodes_list_xy_rooms = []
        self.nodes_list_middle_doors = []
        self.nodes_list_stair_points = []
        self.nodes_list_exit = []
        self.final_transformed_paths = []
        self.middle_nodes_lines = {}
        self.source_node_counts = []
        self.source_node_start_index_of_each_floor = []
        self.link_information_array = None
        self.result_csv_array = None
        self.num_of_source_node_array = None
        self.sub_route_dxdy = []
        self.processed_images = []
        self.final_result = None
        self.movement_paths = []
        self.progress_time = -0.1
        self.Trans = None
        self.final_height = 0


    def find_continuous_indexes(self, array):
        continuous_indexes = {}
        start_index = None
        for i, num in enumerate(array):
            if num > 100000:
                if start_index is None:
                    start_index = i
            elif start_index is not None:
                end_index = i - 1
                if end_index - start_index + 1 >= 3:
                    value = array[start_index] // 100000
                    continuous_indexes[value] = (start_index, end_index)
                start_index = None

        if start_index is not None:
            end_index = len(array) - 1
            if end_index - start_index + 1 >= 3:
                value = array[start_index] // 100000
                continuous_indexes[value] = (start_index, end_index)

        return continuous_indexes if continuous_indexes else None

    def find_index(self, arr, value):
        for i in range(len(arr)):
            if arr[i] == value:
                return i
        return -1

    def rewrite_lists(self, lists, n):
        return [(n, coord) for coord in lists]

    def update(self, frame):
        # 清除之前的文本标签
        for txt in self.ax.texts:
            if txt != self.progress_text:
                txt.set_visible(False)

        for i, path in enumerate(self.movement_paths):
            if frame < len(path):
                current_position = path[frame]
                if current_position == (-1, -1):
                    self.markers[i].set_data([], [])
                else:
                    self.markers[i].set_color('#663300')
                    self.markers[i].set_data([current_position[0]], [current_position[1]])
            else:
                self.markers[i].set_data([], [])

        # 更新进度条文本
        self.progress_text.set_text(f'Evacuation Time: {self.progress_time:.1f}')
        self.progress_time += 0.1
        self.progress_time = round(self.progress_time, 1)
        print('当前计算的疏散时刻：', self.progress_time)
        return [self.markers, self.progress_text]

    def check_consecutive_positions(self, path, current_frame):
        consecutive_count = 1
        for i in range(current_frame - 1, -1, -1):
            if path[i] == path[current_frame]:
                consecutive_count += 1
            else:
                break
        for i in range(current_frame + 1, len(path)):
            if path[i] == path[current_frame]:
                consecutive_count += 1
            else:
                break
        return consecutive_count >= 5

    def find_negative_index(self, array, row_index):
        row = array[row_index]
        negative_index = np.where(row == -1)[0]
        return negative_index[0] if len(negative_index) > 0 else None

    def find_link_value(self, link_information, t_start_index, t_end_index):
        for row in link_information:
            if row[1] == t_start_index and row[2] == t_end_index:
                return row[0]
        return None

    def find_first_and_last_index(self, arr, value):
        first_index = -1
        last_index = -1

        for index in range(len(arr)):
            if arr[index] == value:
                if first_index == -1:
                    first_index = index
                last_index = index
        return (first_index, last_index)

    def interpolate_trajectory(self, points, new_point_count):
        if new_point_count <= 0:
            return points

        new_trajectory = []
        remaining_points = new_point_count - 2

        if remaining_points <= 0:
            return [points[0], points[-1]]

        total_distance = sum(
            ((points[i][0] - points[i - 1][0]) ** 2 + (points[i][1] - points[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(points))
        )

        current_distance = 0

        for idx in range(1, len(points)):
            distance = ((points[idx][0] - points[idx - 1][0]) ** 2 + (points[idx][1] - points[idx - 1][1]) ** 2) ** 0.5
            current_distance += distance

            while len(new_trajectory) < new_point_count - 1 and current_distance >= total_distance / (
                    new_point_count - 1):
                ratio = (current_distance - total_distance / (new_point_count - 1)) / distance
                new_x = round(points[idx - 1][0] + ratio * (points[idx][0] - points[idx - 1][0]))
                new_y = round(points[idx - 1][1] + ratio * (points[idx][1] - points[idx - 1][1]))
                new_trajectory.append((new_x, new_y))
                remaining_points -= 1
                current_distance -= total_distance / (new_point_count - 1)

            if remaining_points <= 0:
                return [points[0]] + new_trajectory + [points[-1]]

        return [points[0]] + new_trajectory + [points[-1]]

    def find_first_occurrence_index(self, arr1, arr2):
        for i, item1 in enumerate(arr1):
            if item1 in arr2:
                return i
        return -1

    def remove_closest_busy_(self, path, col):
        index = col
        is_removed = False
        while index < len(path) - 2 and not is_removed:
            if path[index] == path[index + 1] == path[index + 2]:
                del path[index + 1]
                is_removed = True
            else:
                index += 1
        return path, is_removed

    def calculate_busy_nodes_people_num(self, result_array):
        busy_nodes_people_num = []

        for col_index in range(1, result_array.shape[1]):
            col = result_array[:, col_index]
            counts = {}
            for value in col:
                if value > 100000:
                    counts[value] = counts.get(value, 0) + 1

            busy_nodes = [[key / 100000, value] for key, value in counts.items()]
            busy_nodes_people_num.append(busy_nodes if busy_nodes else [])

        return busy_nodes_people_num

    def translate_path_to_n(self, path, n, dxdy_list, obstacle_map_array=None, person_v=4):
        paths = []
        path_1 = []
        path_2 = []
        around_start_points = []

        random.shuffle(dxdy_list)
        for point in path:
            if point == (-1, -1):
                path_1.append(point)
            else:
                path_2.append(point)

        paths.append(path_2)
        around_start_points.append(path_2[0])
        count_lines = 1

        for dx, dy in dxdy_list:
            n_path = [(x + dx, y + dy) for x, y in path_2]
            if self.check_trajectory(obstacle_map_array, n_path, person_v):
                if count_lines < n:
                    paths.append(n_path)
                    count_lines += 1
                else:
                    break

        for dx, dy in dxdy_list:
            n_path = [(x + dx, y + dy) for x, y in path_2]
            if self.check_trajectory(obstacle_map_array, n_path, 1):
                if count_lines < n:
                    paths.append(n_path)
                    count_lines += 1
                else:
                    break

        if count_lines < n:
            print("未能找到子路径")

        source_node_people_points_quene = self.generate_points(90, 16, 15)
        random.shuffle(source_node_people_points_quene)

        for _ in range(n - 1):
            dx, dy = source_node_people_points_quene[_]
            n_point = (path_2[0][0] + dx, path_2[0][1] + dy)
            around_start_points.append(n_point)

        final_paths = self.match_points_to_paths(around_start_points, paths)

        if path_1:
            return [path_1 + row for row in final_paths]
        else:
            return final_paths

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_avg_distance(self, points):
        if len(points) <= 1:
            return 0
        distances = [self.calculate_distance(points[i], points[i + 1]) for i in range(len(points) - 1)]
        return np.mean(distances)

    def calculate_manhattan_distances(self, movement_paths, threshold):
        num_paths = len(movement_paths)
        result = []

        for i in range(num_paths):
            path_result = []
            for t in range(len(movement_paths[i])):
                current_point = movement_paths[i][t]
                count = 0
                for j in range(num_paths):
                    if j != i and t < len(movement_paths[j]):
                        point_to_compare = movement_paths[j][t]
                        distance = abs(current_point[0] - point_to_compare[0]) + abs(
                            current_point[1] - point_to_compare[1])
                        if distance < threshold:
                            count += 1
                path_result.append(count)
            result.append(path_result)

        return result

    def interpolate_points(self, start_point, end_point, avg_distance):
        distance = self.calculate_distance(start_point, end_point)
        if distance == 0:
            return []

        num_points = int(distance / avg_distance)
        if num_points <= 0:
            return []

        delta_x = (end_point[0] - start_point[0]) / (num_points + 1)
        delta_y = (end_point[1] - start_point[1]) / (num_points + 1)
        return [(start_point[0] + i * delta_x, start_point[1] + i * delta_y) for i in range(1, num_points + 1)]

    def generate_points(self, l, n, threshold):
        points = []
        while len(points) < n:
            x = random.uniform(-l / 2, l / 2)
            y = random.uniform(-l / 2, l / 2)
            point = (x, y)
            valid = True
            for existing_point in points:
                dx = existing_point[0] - x
                dy = existing_point[1] - y
                if math.sqrt(dx ** 2 + dy ** 2) < threshold:
                    valid = False
                    break
            if valid:
                points.append(point)
        return points

    def match_points_to_paths(self, points, paths):
        num_points = len(points)
        num_paths = len(paths)
        distance_matrix = np.zeros((num_points, num_paths))

        for i, point in enumerate(points):
            for j, path in enumerate(paths):
                distance_matrix[i, j] = self.calculate_distance(point, path[0])

        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        matched_pairs = [(points[i], paths[j]) for i, j in zip(row_indices, col_indices)]

        avg_distance = self.calculate_avg_distance(paths[0][1:10]) if len(paths[0]) > 1 else 1

        for point, path in matched_pairs:
            interpolated_points = self.interpolate_points(point, path[0], avg_distance)
            path[0:0] = interpolated_points

        return paths

    def check_trajectory(self, array, trajectory, r):
        if array is None:
            return True

        for point in trajectory:
            nx, ny = int(point[0]), int(point[1])
            check_points = [(nx + x, ny + y) for x in range(-r, r + 1) for y in range(-r, r + 1) if abs(x) + abs(y) < r]
            for px, py in check_points:
                if array[py, px] == 1:
                    return False
        return True

    def add_noise_to_trajectory(self, trajectory, mean=0, std_dev=4, noise_frequency=0.5):
        noisy_trajectory = []
        for point in trajectory:
            if point == (-1, -1):
                noisy_trajectory.append(point)
            else:
                T_F = 1
                while T_F:
                    x_noise = random.gauss(mean, std_dev)
                    y_noise = random.gauss(mean, std_dev)
                    if abs(x_noise) <= std_dev and abs(y_noise) <= std_dev:
                        noisy_point = (point[0] + x_noise, point[1] + y_noise)
                        T_F = 0
                noisy_trajectory.append(noisy_point)
        return noisy_trajectory

    def smooth_trajectory(self, trajectory, window_size):
        if not trajectory:
            return []

        smoothed_trajectory_1 = []
        smoothed_trajectory_2 = []

        for point in trajectory:
            if point == (-1, -1):
                smoothed_trajectory_1.append(point)
            else:
                smoothed_trajectory_2.append(point)

        if not smoothed_trajectory_2:
            return smoothed_trajectory_1

        num_points = len(smoothed_trajectory_2)
        half_window = window_size // 2
        smoothed_points = []

        for i in range(num_points):
            start = max(0, i - half_window)
            end = min(num_points, i + half_window + 1)
            window_size_actual = end - start
            window = smoothed_trajectory_2[start:end]
            smoothed_point = (
                sum(x for x, _ in window) / window_size_actual,
                sum(y for _, y in window) / window_size_actual
            )
            smoothed_points.append(smoothed_point)

        return smoothed_trajectory_1 + smoothed_points

    def remove_duplicates(self, lst):
        seen = set()
        result = []
        for sub_lst in lst:
            if tuple(sub_lst) not in seen:
                result.append(sub_lst)
                seen.add(tuple(sub_lst))
        return result

    def linear_interpolate(self, start, end, num_points):
        if num_points <= 0:
            return []
        x_values = np.linspace(start[0], end[0], num_points + 2)
        y_values = np.linspace(start[1], end[1], num_points + 2)
        return [(x, y) for x, y in zip(x_values[1:-1], y_values[1:-1])]

    def interpolate_path(self, path):
        new_path = []
        i = 0
        while i < len(path):
            if path[i] != (-1, -1):
                new_path.append(path[i])
                i += 1
            else:
                start_idx = i
                while i < len(path) and path[i] == (-1, -1):
                    i += 1
                end_idx = i
                if start_idx > 0 and end_idx < len(path):
                    start_point = path[start_idx - 1]
                    end_point = path[end_idx]
                    num_interpolations = end_idx - start_idx
                    interpolated_points = self.linear_interpolate(start_point, end_point, num_interpolations)
                    new_path.extend(interpolated_points)
                else:
                    new_path.extend(path[start_idx:end_idx])
        return new_path

    def find_first_non_negative_point(self, path):
        for p_id, p in enumerate(path):
            if p != (-1, -1):
                return p_id, p
        return len(path), (-1, -1)

    def find_and_replace_path(self, paths, path):
        best_path = None
        best_path_index = None
        min_non_neg_index = float('inf')

        for i_num, tp in enumerate(paths):
            if len(tp) > len(path) and tp[len(path)] == (-1, -1):
                p_id, first_non_neg_p = self.find_first_non_negative_point(tp)
                if abs(first_non_neg_p[0] - path[-1][0]) < 50:
                    if p_id < min_non_neg_index:
                        min_non_neg_index = p_id
                        best_path = tp
                        best_path_index = i_num

        if best_path is not None:
            for tid in range(len(path)):
                best_path[tid] = path[tid]
            return best_path_index

        return None

    def draw_human_distribution(self, image, human_coords):
        for coord in human_coords:
            cv2.circle(image, coord, 5, (42, 42, 165), -1)

    def load_data(self):
        # 获取所有在该文件夹内的单层文件夹名称列表
        dirs_names = [f for f in os.listdir(self.dirs_path)]
        exclude_files = ['Output.xls', 'Output.xlsx', 'result.csv', 'stairs.txt', 'min_path.txt', 'process.mp4',
                         'multi_result.png']
        dirs_names = [f for f in dirs_names if f not in exclude_files]

        # 存储每层门节点的具体节点信息
        for dir_num in range(len(dirs_names)):
            setattr(self, f'xy_doors_middle_lines_{dir_num + 1}', [])
            setattr(self, f'middle_doors_rects_{dir_num + 1}', [])
            setattr(self, f'mmap_array_inv_{dir_num + 1}', [])

        # 获取多层所有节点的序列
        for dir_num in range(len(dirs_names)):
            fold_dir = os.path.join(self.dirs_path, dirs_names[dir_num])

            input_path = {
                'stair_points': os.path.join(self.dirs_path, 'stairs.txt'),
                'origin': os.path.join(fold_dir, '2opening/seg.png'),
                'draw': os.path.join(fold_dir, 'rgroute.png'),
                'xy_doors': os.path.join(fold_dir, '3approxPOLY/xy_doors.txt'),
                'xy_rooms': os.path.join(fold_dir, '3approxPOLY/xy_rooms.txt'),
                'exit_doors': os.path.join(fold_dir, '6connection/exit_doors.txt'),
                'path_points': os.path.join(fold_dir, '4route/path_points.txt'),
                'middle_doors': os.path.join(fold_dir, '4route/middle_doors.txt'),
                'normal_vector': os.path.join(fold_dir, '5width/normal_vector.txt'),
                'width': os.path.join(fold_dir, '5width/width.txt'),
                'exit_doors_corners': os.path.join(fold_dir, '4route/exit_doors_areas'),
                'map': os.path.join(fold_dir, 'route.png'),
                'route_3': os.path.join(fold_dir, '2opening/route_3.png'),
                'min_path': os.path.join(self.dirs_path, 'min_path.txt'),
                'xy_doors_middle_lines': os.path.join(fold_dir, '4route/xy_doors_middle_lines.txt'),
                'middle_doors_rects': os.path.join(fold_dir, '4route/middle_doors_rects.txt'),
                'output': os.path.join(self.dirs_path, 'Output.xlsx'),
                'result': os.path.join(self.dirs_path, 'result.csv'),
            }

            # 初始化地图
            image = cv2.imread(input_path['origin'], cv2.IMREAD_GRAYSCALE)
            wall = cv2.imread(input_path['route_3'], cv2.IMREAD_GRAYSCALE)
            threshold = 128
            binary_map_inv = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY_INV)[1]
            binary_wall = cv2.threshold(wall, threshold, 1, cv2.THRESH_BINARY)[1]
            final_map = cv2.bitwise_or(binary_map_inv, binary_wall)
            setattr(self, f'mmap_array_inv_{dir_num + 1}', np.array(final_map))

            # 读取门、房间、出口、中间节点坐标
            xy_rooms = self.read_txt_file(input_path['xy_rooms'])
            middle_doors = self.read_txt_file(input_path['middle_doors'])
            exit_doors = self.read_txt_file(input_path['exit_doors'])

            # 读取楼梯点
            with open(input_path['stair_points'], "r") as file:
                lines = file.readlines()
                stair_points = []
                for line in lines:
                    stair_point = line.strip()[1:-1].split('), (')
                    stair_point = [tuple(map(int, n.split(','))) for n in stair_point]
                    stair_points.append(stair_point)
                setattr(self, 'stair_points', stair_points)

            # 读取所有门节点的较长的中间线
            xy_doors_middle_lines = []
            with open(input_path['xy_doors_middle_lines'], 'r') as file:
                for line in file:
                    tuples = eval(line.strip())
                    xy_doors_middle_lines.append(tuples)
            setattr(self, f'xy_doors_middle_lines_{dir_num + 1}', xy_doors_middle_lines)

            # 重写列表并添加到节点列表
            temp1 = self.rewrite_lists(xy_rooms, dir_num + 1)
            temp2 = self.rewrite_lists(middle_doors, dir_num + 1)
            temp4 = self.rewrite_lists(stair_points[dir_num], dir_num + 1)
            temp5 = self.rewrite_lists(exit_doors, dir_num + 1)

            self.nodes_list_xy_rooms += temp1
            self.nodes_list_middle_doors += temp2
            self.nodes_list_stair_points += temp4
            if dir_num == 0:
                self.nodes_list_exit += temp5

        self.nodes_list = (self.nodes_list_xy_rooms + self.nodes_list_middle_doors +
                           self.nodes_list_stair_points + self.nodes_list_exit)

        # 处理xy_doors_middle_lines，仅保留中间节点
        for middle_node in self.nodes_list_middle_doors:
            t_dir_num = middle_node[0]
            not_found = True
            for one_line_details in getattr(self, f'xy_doors_middle_lines_{t_dir_num}'):
                if middle_node[1] in one_line_details:
                    self.middle_nodes_lines[middle_node] = one_line_details
                    not_found = False

            if not_found:
                for error in range(-2, 3):
                    for one_line_details in getattr(self, f'xy_doors_middle_lines_{t_dir_num}'):
                        if (middle_node[1][0] + error, middle_node[1][1] + error) in one_line_details:
                            self.middle_nodes_lines[middle_node] = one_line_details
                            not_found = False
                if not_found:
                    print("有中间节点未能匹配：", middle_node[1])

        # 读取CSV结果文件
        df = pd.read_csv(input_path['result'], encoding='gbk', header=None)
        array_rows = []
        for col in df.columns:
            array_rows.append(df[col].tolist())
        self.result_csv_array = np.array(array_rows)

        # 读取源节点人数信息
        sheet_name_02 = 'Number of Source Nodes'
        df_num = pd.read_excel(input_path['output'], sheet_name_02, engine='openpyxl')
        self.num_of_source_node_array = df_num.values

        # 统计建筑层数与每层源节点人数
        self.source_node_counts = [0 for _ in range(len(dirs_names))]
        for q in range(len(self.num_of_source_node_array)):
            t_node_floor = self.nodes_list[q][0] - 1
            self.source_node_counts[t_node_floor] += 1

        self.source_node_start_index_of_each_floor = []
        sum_of_source_nodes = 0
        for q in range(len(self.source_node_counts)):
            if q > 0:
                sum_of_source_nodes += self.source_node_counts[q - 1]
            self.source_node_start_index_of_each_floor.append(sum_of_source_nodes)

        # 读取节点间路径列表信息
        sheet_name_01 = 'Link Information'
        df_num = pd.read_excel(input_path['output'], sheet_name_01, engine='openpyxl')
        self.link_information_array = df_num.values

        # 子路径衍生矩阵
        interval_distance_1 = 3
        self.sub_route_dxdy = [(a, b) for a in range(-30, 30, interval_distance_1) for b in
                               range(-30, 30, interval_distance_1)]
        if (0, 0) in self.sub_route_dxdy:
            self.sub_route_dxdy.remove((0, 0))

        return dirs_names

    def read_txt_file(self, file_path):
        data = []
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                line = f.readline()
                while line:
                    txt_data = eval(line)
                    data.append(txt_data)
                    line = f.readline()
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
        return data

    def generate_merged_map(self, dirs_names):
        original_points_to_connect = []
        for q in range(len(getattr(self, 'stair_points')[0])):
            temp_points = []
            for p in range(len(getattr(self, 'stair_points'))):
                temp_points.append(getattr(self, 'stair_points')[p][q])
            original_points_to_connect.append(temp_points)

        transformed_points = []
        self.Trans = None
        for current_floor_num in range(1, len(dirs_names) + 1):
            image_path = os.path.join(self.dirs_path, f'{current_floor_num}/route.png')
            img = cv2.imread(image_path)
            cv2.rectangle(img, (100, 100), (img.shape[1] - 100, img.shape[0] - 100), (0, 0, 0), thickness=3)

            # 定义源点和目标点
            AffinePoints0 = np.array([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]],
                                     dtype=np.float32)

            lean_ratio = 0.3
            scale_ratio = 0.75
            img_width = img.shape[1]
            img_height = img.shape[0]
            lean_length = lean_ratio * img_width

            AffinePoints1 = np.array([
                [lean_length, 0],
                [img_width - lean_length, 0],
                [lean_length * (1 - scale_ratio), img_height * (1 - scale_ratio)],
                [img_width - lean_length * (1 - scale_ratio), img_height * (1 - scale_ratio)]
            ], dtype=np.float32)

            # 计算透视变换矩阵
            Trans = cv2.getPerspectiveTransform(AffinePoints0, AffinePoints1)
            self.Trans = Trans
            # 应用透视变换到原始图像
            dst_perspective = cv2.warpPerspective(img, Trans, (img.shape[1], img.shape[0]))

            # 创建白色背景的图像
            white_background = np.full_like(img, 255)
            transformed_background = cv2.warpPerspective(white_background, Trans, (img.shape[1], img.shape[0]))

            # 使用掩码，将透视变换后背景中的黑色部分替换为白色
            mask = np.all(transformed_background == [0, 0, 0], axis=-1)
            dst_perspective[mask] = [255, 255, 255]

            # 裁剪图像
            roi = dst_perspective[0:int(img_height * (1 - scale_ratio)), 0:int(img_width)]
            self.processed_images.append(roi)

        # 处理所有图像后，将它们上下拼接在一起
        self.processed_images.reverse()
        final_points_to_connect = []
        for t_connect_points in original_points_to_connect:
            t_connect_points.reverse()
            final_points_to_connect.append(t_connect_points)

        self.final_result = np.vstack(self.processed_images)
        final_height = self.final_result.shape[0]
        self.final_height = final_height
        # 计算透视变换后的点的坐标
        for points in final_points_to_connect:
            transformed_points.append(cv2.perspectiveTransform(np.array([points], dtype=np.float32), Trans)[0])

        for p in range(len(transformed_points)):
            for q in range(len(transformed_points[p])):
                transformed_points[p][q][1] += final_height / len(transformed_points[p]) * q

        transformed_points = np.round(transformed_points).astype(int)

        # 连接纵向节点
        colors = [(255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                  (31, 119, 180), (140, 86, 75), (227, 119, 194), (127, 127, 127),
                  (188, 189, 34), (23, 190, 207)]

        for points in transformed_points:
            points_for_cv2 = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            for num in range(len(points) - 1):
                color_index = num % len(colors)
                color = colors[color_index]
                overlay = self.final_result.copy()
                cv2.line(overlay, tuple(points[num]), tuple(points[num + 1]), color, thickness=15)
                alpha = 150
                cv2.addWeighted(overlay, alpha / 255.0, self.final_result, 1 - alpha / 255.0, 0, self.final_result)

        output_result_path = os.path.join(self.dirs_path, "multi_result.png")
        cv2.imwrite(output_result_path, self.final_result)
        return self.final_result

    def process_floor_paths(self, current_floor_num, dirs_names):
        print('当前计算楼层：', current_floor_num)

        # 计算min_path_nodes_xy
        min_path_nodes_xy = []
        min_path_indexes_current_floor = []

        # 读取最短路
        min_path_file = os.path.join(self.dirs_path, 'min_path.txt')
        min_path_indexes = []
        with open(min_path_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                t_index = [int(num) for num in line.strip('[]\n').split(',')]
                min_path_indexes.append(t_index)

        for t_min_path in min_path_indexes:
            min_path_node_xy = []
            min_path_indexes_now = []
            is_none = 0
            for t_path_node in t_min_path:
                node_floor_num = self.nodes_list[t_path_node][0]
                if node_floor_num == current_floor_num:
                    min_path_node_xy.append(self.nodes_list[t_path_node][1])
                    min_path_indexes_now.append(t_path_node)
                    is_none = 1
            if is_none:
                min_path_nodes_xy.append(min_path_node_xy)
                min_path_indexes_current_floor.append(min_path_indexes_now)

        # 找出第一层最短路的对应的具体路径
        detail_paths = []
        path_details_file = os.path.join(self.dirs_path, f'{current_floor_num}/4route/path_details_1.txt')
        with open(path_details_file, 'r') as file:
            for line in file:
                point_list = eval(line.strip())
                detail_paths.append(point_list)

        # 从所有路中筛选出对应的最短路
        min_path_details = []
        for one_min_path_nodes in min_path_nodes_xy:
            for one_detail_path in detail_paths:
                if one_detail_path[0] == one_min_path_nodes[0] and one_detail_path[-1] == one_min_path_nodes[-1]:
                    min_path_details.append(one_detail_path)

        # 处理该层对应的源节点路径
        min_path_details_3 = []
        for i in range(self.source_node_counts[current_floor_num - 1]):
            if current_floor_num == 1:
                one_min_path_details = min_path_details[i]
                source_node_number = self.source_node_start_index_of_each_floor[current_floor_num - 1] + i
                continuous_indexes = self.find_continuous_indexes(self.result_csv_array[source_node_number])

                if continuous_indexes is None:
                    t_path = self.interpolate_trajectory(one_min_path_details,
                                                         self.find_negative_index(self.result_csv_array,
                                                                                  source_node_number))
                    min_path_details_3.append(t_path)
                else:
                    keys = list(continuous_indexes.keys())
                    t_divide_point_idx = 0
                    t_time_idx_last = 0
                    t_path = []

                    for k in keys:
                        t_idx_start = continuous_indexes[k][0]
                        t_idx_end = continuous_indexes[k][1]
                        one_middle_node_line = self.middle_nodes_lines[self.nodes_list[int(k)]]
                        first_cross_point_index = self.find_first_occurrence_index(one_min_path_details,
                                                                                   one_middle_node_line)

                        t_part_of_path_1 = self.interpolate_trajectory(
                            one_min_path_details[t_divide_point_idx:first_cross_point_index],
                            t_idx_start - t_time_idx_last - 1
                        )

                        t_part_of_path_2 = [one_min_path_details[first_cross_point_index] for _ in
                                            range(t_idx_start, t_idx_end + 1)]

                        t_path += t_part_of_path_1 + t_part_of_path_2
                        t_divide_point_idx = first_cross_point_index + 1
                        t_time_idx_last = t_idx_end

                    t_part_of_path_3 = self.interpolate_trajectory(
                        one_min_path_details[t_divide_point_idx:],
                        self.find_negative_index(self.result_csv_array, source_node_number) - t_idx_end
                    )
                    t_path += t_part_of_path_3
                    min_path_details_3.append(t_path)
            else:
                one_min_path_details = min_path_details[i]
                source_node_number = self.source_node_start_index_of_each_floor[current_floor_num - 1] + i
                t_index_paths = min_path_indexes_current_floor[i]
                continuous_indexes = self.find_continuous_indexes(self.result_csv_array[source_node_number])

                if continuous_indexes is not None:
                    keys = list(continuous_indexes.keys())
                    for k in keys:
                        if k not in t_index_paths:
                            del continuous_indexes[k]

                if continuous_indexes is None:
                    t_start_index = t_index_paths[-2]
                    t_end_index = t_index_paths[-1]
                    link_value = self.find_link_value(self.link_information_array, t_start_index, t_end_index)
                    t_idx_start, t_idx_end = self.find_first_and_last_index(self.result_csv_array[source_node_number],
                                                                            link_value)
                    t_path = self.interpolate_trajectory(one_min_path_details, t_idx_end)
                    min_path_details_3.append(t_path)
                else:
                    keys = list(continuous_indexes.keys())
                    if t_index_paths[-1] not in keys:
                        t_divide_point_idx = 0
                        t_time_idx_last = 0
                        t_path = []

                        for k in keys:
                            t_idx_start = continuous_indexes[k][0]
                            t_idx_end = continuous_indexes[k][1]
                            one_middle_node_line = self.middle_nodes_lines[self.nodes_list[int(k)]]
                            first_cross_point_index = self.find_first_occurrence_index(one_min_path_details,
                                                                                       one_middle_node_line)

                            t_part_of_path_1 = self.interpolate_trajectory(
                                one_min_path_details[t_divide_point_idx:first_cross_point_index],
                                t_idx_start - t_time_idx_last - 1
                            )

                            t_part_of_path_2 = [one_min_path_details[first_cross_point_index] for _ in
                                                range(t_idx_start, t_idx_end + 1)]

                            t_path += t_part_of_path_1 + t_part_of_path_2
                            t_divide_point_idx = first_cross_point_index + 1
                            t_time_idx_last = t_idx_end

                        t_start_index = t_index_paths[-2]
                        t_end_index = t_index_paths[-1]
                        link_value = self.find_link_value(self.link_information_array, t_start_index, t_end_index)
                        t_idx_start, t_idx_end = self.find_first_and_last_index(
                            self.result_csv_array[source_node_number], link_value)

                        t_part_of_path_3 = self.interpolate_trajectory(
                            one_min_path_details[t_divide_point_idx:],
                            t_idx_end - t_idx_start
                        )
                        t_path += t_part_of_path_3
                        min_path_details_3.append(t_path)
                    elif t_index_paths[-1] in keys:
                        if len(keys) == 1:
                            t_start_index = t_index_paths[-2]
                            t_end_index = t_index_paths[-1]
                            link_value = self.find_link_value(self.link_information_array, t_start_index, t_end_index)
                            t_idx_start, t_idx_end = self.find_first_and_last_index(
                                self.result_csv_array[source_node_number], link_value)
                            t_path = self.interpolate_trajectory(one_min_path_details, t_idx_end)
                            min_path_details_3.append(t_path)
                        else:
                            t_divide_point_idx = 0
                            t_time_idx_last = 0
                            t_path = []

                            for k in keys[:-1]:
                                t_idx_start = continuous_indexes[k][0]
                                t_idx_end = continuous_indexes[k][1]
                                one_middle_node_line = self.middle_nodes_lines[self.nodes_list[int(k)]]
                                first_cross_point_index = self.find_first_occurrence_index(one_min_path_details,
                                                                                           one_middle_node_line)

                                t_part_of_path_1 = self.interpolate_trajectory(
                                    one_min_path_details[t_divide_point_idx:first_cross_point_index],
                                    t_idx_start - t_time_idx_last - 1
                                )

                                t_part_of_path_2 = [one_min_path_details[first_cross_point_index] for _ in
                                                    range(t_idx_start, t_idx_end + 1)]

                                t_path += t_part_of_path_1 + t_part_of_path_2
                                t_divide_point_idx = first_cross_point_index + 1
                                t_time_idx_last = t_idx_end

                            t_part_of_path_3 = self.interpolate_trajectory(
                                one_min_path_details[t_divide_point_idx:],
                                continuous_indexes[keys[-1]][1] - t_idx_end
                            )
                            t_path += t_part_of_path_3
                            min_path_details_3.append(t_path)

        # 处理上层来到该层的节点路径
        if current_floor_num == 1:
            for i in range(self.source_node_counts[current_floor_num - 1], len(min_path_details)):
                one_min_path_details = min_path_details[i]
                source_node_number = self.source_node_start_index_of_each_floor[current_floor_num - 1] + i
                t_index_paths = min_path_indexes_current_floor[i]

                for q in range(len(t_index_paths) - 1):
                    t_start_index = t_index_paths[q]
                    t_end_index = t_index_paths[q + 1]
                    link_value = self.find_link_value(self.link_information_array, t_start_index, t_end_index)
                    t_idx_start, t_idx_end = self.find_first_and_last_index(self.result_csv_array[source_node_number],
                                                                            link_value)

                t_path_1 = [(-1, -1)] * t_idx_start
                t_path_2 = self.interpolate_trajectory(one_min_path_details, t_idx_end - t_idx_start + 1)
                t_path = t_path_1 + t_path_2
                min_path_details_3.append(t_path)

        # 绘制疏散动画
        movement_paths_1 = min_path_details_3
        movement_paths_2 = []

        for q, path in enumerate(movement_paths_1):
            q_num = \
            self.num_of_source_node_array[self.source_node_start_index_of_each_floor[current_floor_num - 1] + q][1]
            t_paths = self.translate_path_to_n(
                path, q_num, self.sub_route_dxdy,
                obstacle_map_array=getattr(self, f'mmap_array_inv_{current_floor_num}'),
                person_v=3
            )
            for p in t_paths:
                movement_paths_2.append(p)

        # 调整拥堵路段的等候状态，防止聚集
        max_time_index = max(len(sublist) for sublist in movement_paths_2)
        threshold_distance = 8


        steps = 8
        for t in range(0, max_time_index - steps):
            for i in range(0, len(movement_paths_2)):
                if t < len(movement_paths_2[i]) and movement_paths_2[i][t] == (-1, -1):
                    continue
                differ_distance_list = [
                    abs(movement_paths_2[j][t + steps][0] - movement_paths_2[i][t + steps][0]) +
                    abs(movement_paths_2[j][t + steps][1] - movement_paths_2[i][t + steps][1])
                    if t + steps < len(movement_paths_2[j]) and t + steps < len(movement_paths_2[i]) else 99
                    for j in range(len(movement_paths_2))
                ]

                collision_points_at_t = sum(1 for distance in differ_distance_list if distance < threshold_distance)

                if collision_points_at_t > 4:
                    movement_paths_2[i].insert(t + 1, movement_paths_2[i][t])
                    movement_paths_2[i], is_or_not_removed = self.remove_closest_busy_(movement_paths_2[i], t + 2)
                    if not is_or_not_removed:
                        del movement_paths_2[i][t + 1]

        # 添加扰动
        movement_paths_3 = [self.add_noise_to_trajectory(t_path, mean=0, std_dev=5, noise_frequency=0.8) for t_path in
                            movement_paths_2]
        movement_paths_4 = [self.smooth_trajectory(t_path, window_size=4) for t_path in movement_paths_3]

        # 记录人员初始位置
        human_distribution_of_one = [
            (int(movement_paths_4[idx][0][0]), int(movement_paths_4[idx][0][1]))
            for idx in range(len(movement_paths_4)) if movement_paths_4[idx][0][0] != -1
        ]

        # 计算透视变换后的路径坐标
        print('开始计算透视变换后坐标：')
        transformed_paths = []
        for path in movement_paths_4:
            t_trans_path = []
            while path and path[0] == (-1, -1):
                t_trans_path.append(path[0])
                path.pop(0)

            if not path:
                transformed_paths.append(t_trans_path)
                continue

            path_array = np.array([path], dtype=np.float32)
            transformed_path = cv2.perspectiveTransform(path_array, self.Trans)[0]
            t_trans_path_2 = [(transformed_path[t][0], transformed_path[t][1]) for t in range(len(transformed_path))]
            t_trans_path.extend(t_trans_path_2)
            transformed_paths.append(t_trans_path)

        print('计算完毕')
        for p in range(len(transformed_paths)):
            for q in range(len(transformed_paths[p])):
                if transformed_paths[p][q][1] != -1:
                    transformed_paths[p][q] = (
                        transformed_paths[p][q][0],
                        transformed_paths[p][q][1] + self.final_height / 3 * (len(dirs_names) - current_floor_num)
                    )

        for p in range(len(transformed_paths)):
            if current_floor_num == 1:
                self.final_transformed_paths.append(transformed_paths[p])
            else:
                changed_path_index = self.find_and_replace_path(self.final_transformed_paths, transformed_paths[p])
                if changed_path_index is not None:
                    self.final_transformed_paths[changed_path_index] = self.interpolate_path(
                        self.final_transformed_paths[changed_path_index])

        return human_distribution_of_one

    def create_animation(self):
        self.movement_paths = self.final_transformed_paths
        fig, self.ax = plt.subplots()
        fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        im = self.ax.imshow(self.final_result)
        self.ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.progress_text = self.ax.text(0.5, 0.95, '', transform=self.ax.transAxes, color='black',
                                          ha='center', va='center', fontweight='bold')
        self.progress_time = -0.1
        self.markers = [self.ax.plot([], [], marker='o', markersize=0.8, linestyle='', color='black')[0]
                        for _ in self.movement_paths]
        frames = max(len(path) for path in self.movement_paths)
        temp_dir = tempfile.mkdtemp()
        frame_files = []
        for frame in range(frames):
            self.update(frame)
            fig.canvas.draw()
            frame_file = os.path.join(temp_dir, f'frame_{frame:06d}.png')
            fig.savefig(frame_file, dpi=300, bbox_inches='tight', pad_inches=0)
            frame_files.append(frame_file)

        # 使用ffmpeg-python创建视频
        output_path = os.path.join(self.dirs_path, 'process.avi')
        self.create_video_from_frames(frame_files, output_path)
        # 清理临时文件
        for file in frame_files:
            os.remove(file)
        os.rmdir(temp_dir)
        plt.close(fig)


    def create_video_from_frames(self, frame_files, output_path):
        # 确保有帧文件
        if not frame_files:
            return
        # 获取第一帧的尺寸
        width, height = Image.open(frame_files[0]).size
        if height % 2 != 0:
            height += 1
        if width % 2 != 0:
            width += 1
        # 创建输入流
        input_pattern = os.path.join(os.path.dirname(frame_files[0]), 'frame_%06d.png')
        input_stream = ffmpeg.input(input_pattern, framerate=30)
        # 设置输出参数
        output_stream = input_stream.output(
            output_path,
            vcodec='mpeg4',  # 'libx264'
            pix_fmt='yuv420p',
            crf=23,
            preset='medium',
            s=f'{width}x{height}'
        )
        # 执行编码
        output_stream.run(overwrite_output=True)


    def run_simulation(self):
        dirs_names = self.load_data()
        self.generate_merged_map(dirs_names)

        human_distribution = []
        for current_floor_num in range(1, len(dirs_names) + 1):
            human_distribution_of_one = self.process_floor_paths(current_floor_num, dirs_names)
            human_distribution.append(human_distribution_of_one)

        self.create_animation()



# 使用示例
# if __name__ == "__main__":
#     dirs_path = "C:/Users/GuYH/Desktop/test/temp"
#     simulator = MultiFloorEvacuationSimulator(dirs_path)
#     simulator.run_simulation()