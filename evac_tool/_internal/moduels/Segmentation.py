import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import shutil
from deep_learning_models.deeplab import DeeplabV3


def dist(box1, box2):
    return math.sqrt((box2[0] - box1[0]) ** 2 + (box2[1] - box1[1]) ** 2)


def process_floor_plan(folder_path):
    """
    处理楼层平面图图像的主函数
    参数:
        folder_path: 包含原始图像文件的文件夹路径
    """
    # 确保路径格式正确
    folder_path = os.path.normpath(folder_path)

    # 获取所有图像文件
    image_names = [f for f in os.listdir(folder_path)
                   if f.lower().endswith((".jpg", ".png"))]

    # 创建处理文件夹并移动图片
    created_dirs = []
    for name in image_names:
        name_base = os.path.splitext(name)[0]
        directory = os.path.join(folder_path, name_base)
        if not os.path.exists(directory):
            os.makedirs(directory)
            created_dirs.append(name_base)

        source_file = os.path.join(folder_path, name)
        destination_file = os.path.join(directory, name)
        shutil.move(source_file, destination_file)

    # 对每个图像文件夹进行处理
    dirs_names = created_dirs  # 只处理新创建的目录

    # 初始化Deeplab模型 (全局共享)
    deeplab = DeeplabV3()
    name_classes = ["background", "room", "door"]

    for dir_name in dirs_names:
        # 1. 准备目录结构
        process_dir = os.path.join(folder_path, dir_name)
        sub_dirs = ['1origin', '2opening', '3approxPOLY', '4route', '5width', '6connection']
        for sub_dir in sub_dirs:
            path = os.path.join(process_dir, sub_dir)
            if not os.path.exists(path):
                os.makedirs(path)
            if sub_dir == '3approxPOLY':
                os.makedirs(os.path.join(path, 'path'), exist_ok=True)

        # 2. 重命名主图像文件
        main_img_path = os.path.join(process_dir, 'route.png')
        orig_files = [f for f in os.listdir(process_dir) if f.lower().endswith(('.jpg', '.png'))]
        if orig_files:
            os.rename(os.path.join(process_dir, orig_files[0]), main_img_path)

        # 3. 定义文件路径映射
        input_path = {
            'origin': main_img_path,
            'seg_doors': os.path.join(process_dir, '1origin/route_0.png'),
            'seg_rooms': os.path.join(process_dir, '1origin/route_1.png'),
        }

        out_path = {
            'img_door_Dilate': os.path.join(process_dir, '2opening/route_0.png'),
            'img_room_Open': os.path.join(process_dir, '2opening/route_1.png'),
            'img_door_mix_room': os.path.join(process_dir, '2opening/seg.png'),
            'xy_doors': os.path.join(process_dir, '3approxPOLY/xy_doors.txt'),
            'xy_rooms': os.path.join(process_dir, '3approxPOLY/xy_rooms.txt'),
            'approx_doors': os.path.join(process_dir, '3approxPOLY/route_0.png'),
            'approx_rooms': os.path.join(process_dir, '3approxPOLY/route_1.png'),
            'network_graph': os.path.join(process_dir, 'rgroute.png'),
            'room_areas': os.path.join(process_dir, '3approxPOLY/room_areas.txt'),
        }

        # 4. 图像分割处理
        try:
            image = Image.open(main_img_path)
            r_image, r_image_0, r_image_1, r_image_2 = deeplab.detect_image(
                image, count=False, name_classes=name_classes
            )
        except Exception as e:
            print(f"图像分割失败: {e}")
            continue

        # 保存分割结果
        r_image.save(os.path.join(process_dir, '1origin/route.png'))
        r_image_0.save(input_path['seg_doors'])
        r_image_1.save(input_path['seg_rooms'])

        # 5. 形态学操作处理
        try:
            img_room = cv2.imread(input_path['seg_rooms'])
            img_door = cv2.imread(input_path['seg_doors'])
            gray_room = cv2.cvtColor(img_room, cv2.COLOR_BGR2GRAY)
            gray_door = cv2.cvtColor(img_door, cv2.COLOR_BGR2GRAY)

            # 房间处理
            kSize = (3, 3)
            kernel_1 = np.ones(kSize, dtype=np.uint8)
            kernel_2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=np.uint8)
            kernel_3 = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.uint8)

            temp_img_room_Open = cv2.morphologyEx(gray_room, cv2.MORPH_OPEN, kernel_1, iterations=5)
            temp_img_room_Open = cv2.morphologyEx(temp_img_room_Open, cv2.MORPH_OPEN, kernel_2, iterations=5)
            img_room_Open = cv2.morphologyEx(temp_img_room_Open, cv2.MORPH_OPEN, kernel_3, iterations=5)

            # 门处理
            _, binary_image = cv2.threshold(gray_door, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_threshold = 50
            filtered_image = np.copy(gray_door)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < area_threshold:
                    cv2.drawContours(filtered_image, [contour], -1, 0, thickness=cv2.FILLED)

            kernel = np.ones((3, 3), dtype=np.uint8)
            img_door_Dilate = cv2.dilate(filtered_image, kernel, iterations=1)

            # 合并处理结果
            _, binary111 = cv2.threshold(img_door_Dilate, 100, 255, cv2.THRESH_BINARY)
            _, binary222 = cv2.threshold(img_room_Open, 100, 255, cv2.THRESH_BINARY)
            img_door_mix_room = cv2.add(binary111, binary222)

            # 保存中间结果
            cv2.imwrite(out_path['img_door_mix_room'], img_door_mix_room)
            cv2.imwrite(os.path.join(process_dir, '5width', 'seg.png'), img_door_mix_room)
            cv2.imwrite(out_path['img_room_Open'], img_room_Open)
            cv2.imwrite(out_path['img_door_Dilate'], img_door_Dilate)

        except Exception as e:
            print(f"形态学操作失败: {e}")
            continue

        # 6. 多边形近似处理
        try:
            # 房间多边形
            ret, binary = cv2.threshold(img_room_Open, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            xy_rooms = []
            with open(out_path['room_areas'], 'w') as f2:
                for cnt in contours:
                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    M = cv2.moments(approx)
                    cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                    cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                    xy_rooms.append((cX, cY))

                    area = int(cv2.contourArea(cnt))
                    f2.write(f"{area}\n")

            with open(out_path['xy_rooms'], 'w') as f:
                for q in xy_rooms:
                    f.write(f"{q}\n")

            # 门多边形
            ret, binary = cv2.threshold(img_door_Dilate, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            xy_doors = []
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).astype(int)
                XY = tuple(map(round, rect[0]))
                xy_doors.append(XY)

            with open(out_path['xy_doors'], 'w') as f:
                for q in xy_doors:
                    f.write(f"{q}\n")

        except Exception as e:
            print(f"多边形近似失败: {e}")
            continue

        # 7. 绘制网络图
        try:
            img = Image.open(input_path['origin'])
            draw = ImageDraw.Draw(img)
            a = 5  # 标记尺寸

            # 使用安全字体或指定字体路径
            try:
                font = ImageFont.truetype('SIMLI.TTF', 20)
            except:
                font = ImageFont.load_default()

            # 标注门位置
            for q, (x, y) in enumerate(xy_doors):
                draw.polygon([
                    x - a / 2, y - a * np.sqrt(3) / 6,
                    x + a / 2, y - a * np.sqrt(3) / 6,
                    x, y + a / np.sqrt(3)
                ], fill='blue', outline='purple')
                draw.text((x, y), f'{x},{y}', font=font, fill='red')

            # 标注房间位置
            for q, (x, y) in enumerate(xy_rooms):
                draw.rectangle(
                    [x - a / 2, y - a / 2, x + a / 2, y + a / 2],
                    fill='blue', outline='purple', width=5
                )
                draw.text((x, y), f'{x},{y}', font=font, fill='red')

            img.save(out_path['network_graph'])

        except Exception as e:
            print(f"网络图生成失败: {e}")
            continue

        # 8. 门宽及法向量计算
        try:
            img_path = os.path.join(process_dir, '2opening', 'route_0.png')
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            with open(os.path.join(process_dir, '5width', 'width_1.txt'), 'w') as f1, \
                    open(os.path.join(process_dir, '5width', 'width.txt'), 'w') as f2, \
                    open(os.path.join(process_dir, '5width', 'normal_vector.txt'), 'w') as f3:

                f1.write('(X,Y)\t\t(宽,高)\n')

                for cnt in contours:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect).astype(int)

                    # 计算法向量
                    n = box[1] - box[2] if dist(box[0], box[1]) > dist(box[1], box[2]) else box[0] - box[1]
                    f3.write(f"{n}\n")

                    # 记录尺寸
                    f1.write(f"{rect[0]}\t{rect[1]}\n")
                    e, f = rect[1]
                    width = max(int(e), int(f))
                    f2.write(f"{width}\n")

        except Exception as e:
            print(f"门宽计算失败: {e}")
            continue

        # 9. 实例分割转换
        try:
            seg_img_path = os.path.join(process_dir, "2opening", "seg.png")
            contour_img = cv2.imread(seg_img_path)
            img_path = os.path.join(process_dir, "2opening", "route_0.png")
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 生成颜色列表
            colors = [(c, c, c) for c in range(3, 256, 3)]

            for i, cnt in enumerate(contours):
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).astype(int)
                color = colors[i % len(colors)]
                cv2.drawContours(contour_img, [box], -1, color, cv2.FILLED)

            # 保存结果并转换
            contours_img_path = os.path.join(process_dir, '6connection', 'contours_0.png')
            cv2.imwrite(contours_img_path, contour_img)

            img = Image.open(contours_img_path)
            gray_contour_img = img.convert('L')
            gray_contour_img.save(os.path.join(process_dir, '6connection', 'contours_1.png'))

        except Exception as e:
            print(f"门实例分割转换失败: {e}")
            continue

    print("处理完成！")

# # 示例调用
# if __name__ == '__main__':
#     # 指定分析路径
#     analysis_path = "C:/Users/GuYH/Desktop/test/temp"
#
#     # 执行路径分析
#     process_floor_plan(
#         analysis_path
#     )