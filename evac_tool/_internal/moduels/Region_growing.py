import numpy as np
from PIL import Image
from collections import deque

def region_growing(img_path, final_path, area_threshold=3000):
    # 一次性读取图像并转为numpy数组
    image = Image.open(img_path).convert('L')
    width, height = image.size
    img_array = np.array(image)

    # 预定义四个角点坐标
    corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]

    # 初始化数据结构
    visited = np.zeros_like(img_array, dtype=bool)
    room_points = []

    # BFS队列初始化
    queue = deque()

    # 方向：上右下左
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # 遍历所有像素
    for y in range(height):
        for x in range(width):
            if img_array[y, x] == 255 and not visited[y, x]:
                # BFS开始
                queue.append((x, y))
                visited[y, x] = True
                region = []
                has_corner = False

                while queue:
                    cx, cy = queue.popleft()
                    region.append((cx, cy))

                    # 检查是否为角点
                    if (cx, cy) in corners:
                        has_corner = True

                    # 检查四邻域
                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < width and 0 <= ny < height and
                                img_array[ny, nx] == 255 and not visited[ny, nx]):
                            visited[ny, nx] = True
                            queue.append((nx, ny))

                # 检查区域条件并记录
                if not has_corner and len(region) > area_threshold:
                    room_points.extend(region)

    # 创建结果图像
    result = np.zeros((height, width), dtype=np.uint8)
    for x, y in room_points:
        result[y, x] = 128

    # 保存最终结果
    Image.fromarray(result).save(final_path)
    print(f"处理完成，找到{len(room_points)}个有效像素点")


# # 使用优化后的函数
# img_path = r'C:\Users\GuYH\Desktop\test\test4\2.png'
# final_path = r'C:\Users\GuYH\Desktop\test\test4\c_route_1.png'
# region_growing(img_path, final_path)
