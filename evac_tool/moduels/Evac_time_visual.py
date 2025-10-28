import os
import cv2
import numpy as np
import pandas as pd


def process_evacuation_times(dirs_path):
    """
    处理疏散时间并在各层平面图上标注

    参数:
    dirs_path (str): 包含多层建筑数据的文件夹路径
    """
    # 获取有效文件夹列表
    dirs_names = [
        f for f in os.listdir(dirs_path)
        if os.path.isdir(os.path.join(dirs_path, f)) and
           f not in ['Output.xls', 'Output.xlsx', 'result.csv',
                     'stairs.txt', 'min_path.txt',
                     'process.mp4', 'multi_result.png']
    ]
    if not dirs_names:
        print("警告：未找到有效楼层文件夹")
        return

    # 存储节点信息
    nodes_list_xy_rooms = []
    nodes_list_middle_doors = []
    nodes_list_stair_points = []
    nodes_list_exit = []

    # 读取整个建筑的楼梯点
    stairs_file = os.path.join(dirs_path, 'stairs.txt')
    all_stair_points = []
    if os.path.exists(stairs_file):
        with open(stairs_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                stair_point = line.strip()[1:-1].split('), (')
                stair_point = [tuple(map(int, n.split(','))) for n in stair_point]
                all_stair_points.append(stair_point)

    # 确保楼梯点列表长度与楼层数匹配
    if len(all_stair_points) < len(dirs_names):
        for _ in range(len(dirs_names) - len(all_stair_points)):
            all_stair_points.append([])

    # 处理每层楼的节点
    for dir_num, dir_name in enumerate(dirs_names):
        fold_dir = os.path.join(dirs_path, dir_name)

        # 源节点 (xy_rooms)
        xy_rooms_file = os.path.join(fold_dir, '3approxPOLY', 'xy_rooms.txt')
        if os.path.exists(xy_rooms_file):
            with open(xy_rooms_file, "r", encoding='utf-8') as f:
                xy_rooms = [eval(line.strip()) for line in f if line.strip()]
                nodes_list_xy_rooms.extend([(dir_num + 1, coord) for coord in xy_rooms])

        # 中间节点 (middle_doors)
        middle_doors_file = os.path.join(fold_dir, '4route', 'middle_doors.txt')
        if os.path.exists(middle_doors_file):
            with open(middle_doors_file, "r", encoding='utf-8') as f:
                middle_doors = [eval(line.strip()) for line in f if line.strip()]
                nodes_list_middle_doors.extend([(dir_num + 1, coord) for coord in middle_doors])

        # 楼梯节点
        stair_points = all_stair_points[dir_num] if dir_num < len(all_stair_points) else []
        nodes_list_stair_points.extend([(dir_num + 1, coord) for coord in stair_points])

        # 出口节点 (只处理第一层)
        if dir_num == 0:
            exit_doors_file = os.path.join(fold_dir, '6connection', 'exit_doors.txt')
            if os.path.exists(exit_doors_file):
                with open(exit_doors_file, "r", encoding='utf-8') as f:
                    exit_doors = [eval(line.strip()) for line in f if line.strip()]
                    nodes_list_exit.extend([(dir_num + 1, coord) for coord in exit_doors])

    # 计算疏散时间
    evacuation_time = []
    result_file = os.path.join(dirs_path, 'result.csv')
    if os.path.exists(result_file):
        df = pd.read_csv(result_file, encoding='gbk', header=None)
        result_csv_array = np.array([df[col].tolist() for col in df.columns])

        def find_negative_index(array, row_index):
            row = array[row_index]
            negative_index = np.where(row == -1)[0]
            return negative_index[0] if negative_index.size > 0 else None

        evacuation_time = []
        for i in range(len(result_csv_array)):
            idx = find_negative_index(result_csv_array, i)
            if idx is not None:
                evacuation_time.append(int(idx * 0.1))
            else:
                evacuation_time.append(0)

    # 获取每层源节点数量和起始索引
    excel_file = os.path.join(dirs_path, 'Output.xlsx')
    if os.path.exists(excel_file):
        try:
            df_num = pd.read_excel(excel_file, sheet_name='Number of Source Nodes', engine='openpyxl')
            num_of_source_node_array = df_num.values

            # 计算每层源节点数量
            source_node_counts = [0] * len(dirs_names)
            for node_info in num_of_source_node_array:
                if len(node_info) > 0 and node_info[0] < len(nodes_list_xy_rooms):
                    t_node_floor = nodes_list_xy_rooms[node_info[0] - 1][0] - 1
                    if t_node_floor < len(source_node_counts):
                        source_node_counts[t_node_floor] += 1

            # 计算每层起始索引
            source_node_start_index_of_each_floor = []
            cumulative = 0
            for count in source_node_counts:
                source_node_start_index_of_each_floor.append(cumulative)
                cumulative += count
        except Exception as e:
            print(f"读取Excel文件时出错: {e}")
            source_node_start_index_of_each_floor = [0] * len(dirs_names)
    else:
        print(f"警告：未找到Excel文件 {excel_file}")
        source_node_start_index_of_each_floor = [0] * len(dirs_names)

    # 在各层平面图上标注疏散时间
    for dir_num, dir_name in enumerate(dirs_names):
        source_node_coords = []
        fold_dir = os.path.join(dirs_path, dir_name)

        # 读取源节点坐标
        xy_rooms_file = os.path.join(fold_dir, '3approxPOLY', 'xy_rooms.txt')
        if os.path.exists(xy_rooms_file):
            with open(xy_rooms_file, "r", encoding='utf-8') as f:
                source_node_coords = [eval(line.strip()) for line in f if line.strip()]

        # 读取楼层地图
        map_file = os.path.join(fold_dir, 'route.png')
        if not os.path.exists(map_file):
            print(f"警告：{dir_name} 楼层地图文件不存在")
            continue

        time_image = cv2.imread(map_file, cv2.IMREAD_UNCHANGED)  # 读取图片并保留透明通道
        if time_image is None:
            print(f"警告：无法读取 {dir_name} 楼层地图")
            continue

        # ==== 新增的像素处理部分 ====
        # 输出(0,0)位置的像素值
        target_pixel = time_image[0, 0]
        # 检查图像是否有透明通道
        has_alpha = False
        if time_image.ndim == 3 and time_image.shape[2] == 4:
            # BGRA格式 (带透明通道)
            has_alpha = True
            b, g, r, a = cv2.split(time_image)
            time_image_bgr = cv2.merge([b, g, r])
        else:
            # BGR格式 (无透明通道)
            time_image_bgr = time_image.copy()

        # 创建掩膜找到所有相同像素的位置
        target_color = target_pixel[:3] if has_alpha else target_pixel
        mask = np.all(time_image_bgr == target_color, axis=-1)

        # 将所有匹配像素设为白色
        time_image_bgr[mask] = [255, 255, 255]

        # 合并回原始格式
        if has_alpha:
            time_image = cv2.merge([time_image_bgr[:, :, 0],
                                    time_image_bgr[:, :, 1],
                                    time_image_bgr[:, :, 2],
                                    a])
        else:
            time_image = time_image_bgr
        # ==== 结束像素处理部分 ====

        if dir_num >= len(source_node_start_index_of_each_floor):
            continue

        start_index = source_node_start_index_of_each_floor[dir_num]
        end_index = start_index + len(source_node_coords)

        # 设置字体参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        for i, coord in enumerate(source_node_coords):
            t_node_num = start_index + i

            if t_node_num < len(evacuation_time):
                evacuation_time_text = str(evacuation_time[t_node_num])
                point = (int(coord[0]), int(coord[1]))
                color = (0, 0, 0)
                # 计算文本尺寸
                (text_width, text_height), baseline = cv2.getTextSize(
                    evacuation_time_text, font, font_scale, thickness)
                text_x = point[0] - text_width // 2
                text_y = point[1] + text_height // 2
                top_left = (text_x - 5, text_y - text_height - 5)
                bottom_right = (text_x + text_width + 5, text_y + 5)
                cv2.rectangle(time_image, top_left, bottom_right, color, thickness)
                cv2.putText(time_image, evacuation_time_text, (text_x, text_y), font, font_scale, color, thickness,
                            cv2.LINE_AA)
        output_path = os.path.join(fold_dir, 'evacuation_time.png')
        cv2.imwrite(output_path, time_image)


# # 使用示例
# if __name__ == "__main__":
#     dirs_path = "C:/Users/GuYH/Desktop/test/test24"
#     process_evacuation_times(dirs_path)