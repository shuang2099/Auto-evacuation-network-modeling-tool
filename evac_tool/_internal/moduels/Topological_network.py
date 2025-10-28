import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import warnings


def generate_topological_networks(dirs_path):
    dirs_names = [f for f in os.listdir(dirs_path)]
    exclude_files = ['Output.xlsx', 'Output.xls', 'result.csv', 'stairs.txt', 'min_path.txt', 'process.mp4',
                     'multi_result.png']
    dirs_names = [d for d in dirs_names if d not in exclude_files]

    layer_node_counts = []
    stair_points_all = []

    if os.path.exists(os.path.join(dirs_path, "stairs.txt")):
        with open(os.path.join(dirs_path, "stairs.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                stair_point = line.strip()[1:-1].split('), (')
                stair_point = [tuple(map(int, n.split(','))) for n in stair_point]
                stair_points_all.append(stair_point)

    def is_exist(arr, element):
        return element in arr

    def draw_point(draw, point, size=15, outline_width=1, number=None, color='red'):
        outline = color
        draw.ellipse((point[0] - size, point[1] - size, point[0] + size, point[1] + size),
                     outline=outline, width=outline_width)
        if number is not None:
            number_str = str(number)
            font_size = int(size * 1.3)
            try:
                font = ImageFont.truetype("arial.ttf", size=font_size)
            except:
                font = ImageFont.load_default()

            # 使用getbbox替代textsize
            bbox = font.getbbox(number_str)
            text_width = bbox[2] - bbox[0]  # right - left
            text_height = bbox[3] - bbox[1]  # bottom - top

            # 计算文本位置
            text_x = point[0] - text_width / 2
            text_y = point[1] - text_height / 2 - size / 5
            draw.text((text_x, text_y), number_str, fill=color, font=font)

    def draw_arrow_on_image(draw, start_point, end_point, arrow_size=10, color=(128, 138, 135), width=2):
        draw.line((start_point, end_point), fill=color, width=width)
        direction = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        if length == 0:
            return
        direction = (-direction[0] / length, -direction[1] / length)
        two_third_point = ((end_point[0] - start_point[0]) * 2 // 3 + start_point[0],
                           (end_point[1] - start_point[1]) * 2 // 3 + start_point[1])
        perp_direction = (-direction[1], direction[0])
        arrow_size_px = arrow_size
        arrow_length = arrow_size_px / math.sin(math.pi / 3)
        arrow_points = [
            (int(two_third_point[0] + arrow_length * direction[0] + arrow_size_px * perp_direction[0] * math.cos(
                math.pi / 3)),
             int(two_third_point[1] + arrow_length * direction[1] + arrow_size_px * perp_direction[1] * math.cos(
                 math.pi / 3))),
            two_third_point,
            (int(two_third_point[0] + arrow_length * direction[0] - arrow_size_px * perp_direction[0] * math.cos(
                math.pi / 3)),
             int(two_third_point[1] + arrow_length * direction[1] - arrow_size_px * perp_direction[1] * math.cos(
                 math.pi / 3)))
        ]
        if math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2) > 5:
            draw.polygon(arrow_points, fill=color)

    def find_index(arr, value):
        try:
            return arr.index(value)
        except ValueError:
            return -1

    def dist(point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def count_up(start=0):
        n = start
        while True:
            yield n
            n += 1

    def calculate_width(point, n, gray_img_path):
        gray_img = Image.open(gray_img_path)
        size = gray_img.size
        ult = max(size[0], size[1])
        for i in range(1, ult):
            x = int(point[0] + n[0] * i)
            y = int(point[1] + n[1] * i)
            if 0 <= x < gray_img.width and 0 <= y < gray_img.height:
                if gray_img.getpixel((x, y)) == 0:
                    break
        return dist(point, (x, y))

    # Process each layer
    for dir_num, dir_name in enumerate(dirs_names):
        fold_dir = os.path.join(dirs_path, dir_name)
        input_path = {
            'origin': os.path.join(fold_dir, '2opening', 'seg.png'),
            'draw': os.path.join(fold_dir, 'rgroute.png'),
            'xy_doors': os.path.join(fold_dir, '3approxPOLY', 'xy_doors.txt'),
            'xy_rooms': os.path.join(fold_dir, '3approxPOLY', 'xy_rooms.txt'),
            'exit_doors': os.path.join(fold_dir, '6connection', 'exit_doors.txt'),
            'path_points': os.path.join(fold_dir, '4route', 'path_points.txt'),
            'middle_doors': os.path.join(fold_dir, '4route', 'middle_doors.txt'),
            'normal_vector': os.path.join(fold_dir, '5width', 'normal_vector.txt'),
            'width': os.path.join(fold_dir, '5width', 'width.txt'),
            'width_change_points': os.path.join(fold_dir, '5width', 'change_points.txt'),
        }

        # 读取数据
        with open(input_path['path_points'], 'r') as f:
            lines = f.readlines()  # 读取所有行
        points_list = []  # 定义一个空列表存储读取的元组
        for line in lines:  # 遍历每一行
            points = []
            line_list = line.strip().split(')(')  # 将字符串按照括号切割
            for item in line_list:  # 遍历每个元素
                # 将括号和逗号去掉，再将每两个数字组成一个元组，添加到points列表中
                points.append(tuple(map(int, item.replace('(', '').replace(')', '').replace(',', ' ').split())))
            points_list.append(points)
        print(points_list)  # 输出读取到的元组列表

        # Read other data files
        def read_txt_points(file_path):
            points = []
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('(') and line.endswith(')'):
                            try:
                                x, y = line[1:-1].split(',')
                                points.append((int(x), int(y)))
                            except:
                                pass
            return points

        xy_doors = read_txt_points(input_path['xy_doors'])
        xy_rooms = read_txt_points(input_path['xy_rooms'])
        exit_doors = read_txt_points(input_path['exit_doors'])
        middle_doors = read_txt_points(input_path['middle_doors'])
        width_change_points = read_txt_points(input_path['width_change_points'])

        # Store node counts for this layer
        layer_node_counts.append([
            len(xy_rooms),
            len(middle_doors),
            len(stair_points_all[dir_num]) if dir_num < len(stair_points_all) else 0,
            len(exit_doors) if dir_num == 0 else 0
        ])

        # Read width data
        width = []
        if os.path.exists(input_path['width']):
            with open(input_path['width'], 'r') as f:
                for line in f:
                    try:
                        width.append(int(line.strip()))
                    except:
                        pass

        # Process if path points exist
        if points_list:
            img = Image.open(os.path.join(fold_dir, "route.png"))
            draw = ImageDraw.Draw(img)

            # Draw paths and arrows
            for row in points_list:
                for i in range(len(row) - 1):
                    draw_arrow_on_image(draw, row[i], row[i + 1], arrow_size=15, width=1)

            # Calculate OPS values
            need_ops_doors = xy_doors
            door_temps = [[] for _ in need_ops_doors]

            # Identify points leading to each door
            for row in points_list:
                for i in range(len(row) - 1):
                    if row[i + 1] in need_ops_doors:
                        idx = need_ops_doors.index(row[i + 1])
                        door_temps[idx].append(row[i])

            # Read normal vectors
            normal_vectors = []
            if os.path.exists(input_path['normal_vector']):
                with open(input_path['normal_vector'], 'r') as f:
                    for line in f:
                        try:
                            vec = line.strip('[]\n').split()
                            normal_vectors.append((int(vec[0]), int(vec[1])))
                        except:
                            normal_vectors.append((0, 0))

            # Calculate and write OPS
            with open(os.path.join(fold_dir, '5width', 'ops.txt'), 'w') as f:
                for i in range(len(need_ops_doors)):
                    door_width = width[i] if i < len(width) else 10
                    ops = 0.4
                    if is_exist(width_change_points, need_ops_doors[i]):
                        f.write(f"{ops}\n")
                        continue

                    # Calculate door congestion vector
                    n = normal_vectors[i]
                    if n == (0, 0):
                        f.write(f"{ops}\n")
                        continue

                    dot_prod = 0
                    for vec in door_temps[i]:
                        dx = vec[0] - need_ops_doors[i][0]
                        dy = vec[1] - need_ops_doors[i][1]
                        dot_prod += (dx * n[0] + dy * n[1])
                    dot_prod = 1 if dot_prod >= 0 else -1
                    n_len = math.sqrt(n[0] ** 2 + n[1] ** 2)
                    if n_len > 0:
                        unit_vector = (n[0] / n_len * dot_prod, n[1] / n_len * dot_prod)
                    else:
                        unit_vector = (dot_prod, 0)

                    # Calculate passage widths
                    temp_point = (need_ops_doors[i][0] + unit_vector[0] * door_width,
                                  need_ops_doors[i][1] + unit_vector[1] * door_width)
                    try:
                        l0 = calculate_width(temp_point, unit_vector, input_path['origin'])
                        n1 = (-unit_vector[1], unit_vector[0])
                        n2 = (unit_vector[1], -unit_vector[0])
                        l1 = calculate_width(temp_point, n1, input_path['origin'])
                        l2 = calculate_width(temp_point, n2, input_path['origin'])
                        L = l1 + l2
                        scale0 = door_width * 5
                        scale1 = door_width * 10
                        if l0 < scale0 or L < scale1:
                            ops = 0.3
                    except:
                        pass

                    f.write(f"{ops}\n")

            img.save(os.path.join(fold_dir, 'net.png'))

    # Generate node numbering
    starts = [0, 0, 0, 0]
    for counts in layer_node_counts:
        starts[1] += counts[0]
        starts[2] += counts[1]
        starts[3] += counts[2]
    starts[2] += starts[1]
    starts[3] += starts[2]

    counters = [
        count_up(start=starts[0]),  # Origin nodes
        count_up(start=starts[1]),  # Middle nodes
        count_up(start=starts[2]),  # Stair nodes
        count_up(start=starts[3])  # Exit nodes (only for first layer)
    ]

    def count_up(start=0):
        n = start
        while True:
            yield n
            n += 1

    # Number nodes in each layer
    for dir_num, dir_name in enumerate(dirs_names):
        fold_dir = os.path.join(dirs_path, dir_name)
        input_path = {
            'xy_rooms': os.path.join(fold_dir, '3approxPOLY', 'xy_rooms.txt'),
            'exit_doors': os.path.join(fold_dir, '6connection', 'exit_doors.txt'),
            'middle_doors': os.path.join(fold_dir, '4route', 'middle_doors.txt'),
            'net': os.path.join(fold_dir, 'net.png'),
        }

        xy_rooms = read_txt_points(input_path['xy_rooms'])
        exit_doors = read_txt_points(input_path['exit_doors'])
        middle_doors = read_txt_points(input_path['middle_doors'])

        if os.path.exists(input_path['net']):
            img = Image.open(input_path['net'])
            draw = ImageDraw.Draw(img)

            # Draw numbered points
            point_size = 25
            for point in xy_rooms:
                draw_point(draw, point, size=point_size, outline_width=3, number=next(counters[0]), color='black')
            for point in middle_doors:
                draw_point(draw, point, size=point_size, outline_width=3, number=next(counters[1]), color='red')
            if dir_num < len(stair_points_all):
                for point in stair_points_all[dir_num]:
                    draw_point(draw, point, size=point_size, outline_width=3, number=next(counters[2]), color='red')
            if dir_num == 0:
                for point in exit_doors:
                    draw_point(draw, point, size=point_size, outline_width=3, number=next(counters[3]), color='green')

            img.save(input_path['net'])

            # Process background
            if os.path.exists(input_path['net']):
                img_cv = cv2.imread(input_path['net'])
                if img_cv is not None:
                    mask = np.all(img_cv == img_cv[0, 0], axis=2)
                    img_cv[mask] = [255, 255, 255]
                    cv2.imwrite(os.path.join(fold_dir, 'f_net.png'), img_cv)

    # Create multi-layer perspective visualization
    processed_images = []
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
              (148, 103, 189), (140, 86, 75), (227, 119, 194),
              (127, 127, 127), (188, 189, 34), (23, 190, 207)]

    # Create perspective images
    for dir_num, dir_name in enumerate(reversed(dirs_names)):
        fold_dir = os.path.join(dirs_path, dir_name)
        net_path = os.path.join(fold_dir, 'net.png')

        if os.path.exists(net_path):
            img = cv2.imread(net_path)
            if img is not None:
                cv2.rectangle(img, (100, 100), (img.shape[1] - 100, img.shape[0] - 100), (0, 0, 0), 3)

                AffinePoints0 = np.array([
                    [0, 0], [img.shape[1], 0],
                    [0, img.shape[0]], [img.shape[1], img.shape[0]]
                ], dtype=np.float32)

                lean_ratio = 0.3
                scale_ratio = 0.75
                img_width = img.shape[0]  # 图片横向宽度
                img_height = img.shape[1]  # 图片竖向高度
                lean_length = lean_ratio * img_width
                AffinePoints1 = np.array([[lean_length, 0], [img_width - lean_length, 0],
                                          [lean_length * (1 - scale_ratio), img_height * (1 - scale_ratio)],
                                          [img_width - lean_length * (1 - scale_ratio),
                                           img_height * (1 - scale_ratio)]],
                                         dtype=np.float32)

                M = cv2.getPerspectiveTransform(AffinePoints0, AffinePoints1)
                dst = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

                white = np.full_like(img, 255)
                white_dst = cv2.warpPerspective(white, M, (img.shape[1], img.shape[0]))
                mask = np.all(white_dst == [0, 0, 0], axis=-1)
                dst[mask] = [255, 255, 255]

                roi = dst[int(0):int(img_height * (1 - scale_ratio)), 0:int(img_width)]
                processed_images.append(roi)

    # Process stair points
    if stair_points_all and all(len(pts) == len(stair_points_all[0]) for pts in stair_points_all):
        original_points_to_connect = list(zip(*reversed(stair_points_all)))
    else:
        original_points_to_connect = []

    # Connect points in final image
    if processed_images:
        final_result = np.vstack(processed_images)
        final_height = final_result.shape[0]

        transformed_points = []
        for group in original_points_to_connect:
            trans_group = cv2.perspectiveTransform(
                np.array([group], dtype=np.float32), M)[0]
            for idx, pt in enumerate(trans_group):
                pt[1] += final_height / len(group) * idx
            transformed_points.append(np.round(trans_group).astype(int))

        # Draw connections
        for idx, points in enumerate(transformed_points):
            color = colors[idx % len(colors)]
            for j in range(len(points) - 1):
                overlay = final_result.copy()
                cv2.line(overlay, tuple(points[j]), tuple(points[j + 1]), color, 15, cv2.LINE_AA)
                alpha = 0.6
                final_result = cv2.addWeighted(overlay, alpha, final_result, 1 - alpha, 0)

        # Save final result
        output_path = os.path.join(dirs_path, "multi_result.png")
        cv2.imwrite(output_path, final_result)
        return output_path

    return None

# 示例调用
if __name__ == '__main__':
    # 指定分析路径
    analysis_path = "C:/Users/GuYH/Desktop/test/temp"

    # 执行路径分析
    generate_topological_networks(
        dirs_path=analysis_path
     )


