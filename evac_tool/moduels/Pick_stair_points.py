import os

from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QLabel,
                             QTextEdit, QPushButton, QMessageBox, QSizePolicy, QHBoxLayout)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont


class CoordinateRecorderDialog(QDialog):
    def __init__(self, image_paths, output_file, parent=None):
        super().__init__(parent)
        self.setWindowTitle("楼梯节点选择工具")
        # 设置更大的初始窗口尺寸
        self.setMinimumSize(1200, 700)  # 增大窗口尺寸
        self.resize(1200, 800)  # 默认更大的窗口尺寸

        self.image_paths = image_paths
        self.output_file = output_file
        self.current_index = 0
        self.coordinates = []  # 当前图像的坐标
        self.all_coordinates = []  # 所有图像的坐标列表
        self.orig_pixmap = None
        self.scaled_pixmap = None

        # 创建UI元素
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setCursor(Qt.CrossCursor)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFont(QFont("Arial", 12, QFont.Bold))

        self.coords_edit = QTextEdit()
        self.coords_edit.setReadOnly(True)
        self.coords_edit.setFont(QFont("Consolas", 10))
        self.coords_edit.setStyleSheet("font-size: 14px;")

        # 按钮
        btn_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """

        self.prev_btn = QPushButton("上一张")
        self.prev_btn.setStyleSheet(btn_style)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("下一张")
        self.next_btn.setStyleSheet(btn_style)

        self.clear_btn = QPushButton("清除本图")
        self.clear_btn.setStyleSheet(btn_style)

        self.save_exit_btn = QPushButton("保存并退出")
        self.save_exit_btn.setStyleSheet(btn_style)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.image_label, 8)  # 图像标签占据更多空间
        layout.addWidget(self.info_label)
        layout.addWidget(self.coords_edit, 2)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.save_exit_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # 连接信号
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.clear_btn.clicked.connect(self.clear_coordinates)
        self.save_exit_btn.clicked.connect(self.save_and_exit)

        # 加载第一张图片
        self.load_image()

    def load_image(self):
        """加载当前图像并显示"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            QMessageBox.warning(self, "错误", "没有图片可加载")
            self.reject()
            return

        try:
            # 加载原始图像
            image_path = self.image_paths[self.current_index]
            self.orig_pixmap = QPixmap(image_path)

            if self.orig_pixmap.isNull():
                raise ValueError(f"无法加载图片: {image_path}")

            # 获取QLabel的尺寸（减去一些边距）
            label_width = self.image_label.width() - 40
            label_height = self.image_label.height() - 40

            # 如果QLabel的尺寸还没有确定（初始时为0），则使用窗口尺寸估算
            if label_width <= 0 or label_height <= 0:
                label_width = self.width() - 100
                label_height = self.height() - 300

            # 保持宽高比缩放，使用QLabel的尺寸作为最大尺寸
            self.scaled_pixmap = self.orig_pixmap.scaled(
                label_width,
                label_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # 显示图片，并保存缩放比例
            self.image_label.setPixmap(self.scaled_pixmap)

            # 更新按钮状态
            self.update_button_states()

            # 更新信息
            self.update_info()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片失败: {str(e)}")
            self.reject()

    def resizeEvent(self, event):
        """窗口大小改变时重新加载图片"""
        super().resizeEvent(event)
        if hasattr(self, 'orig_pixmap') and self.orig_pixmap:
            self.load_image()


    def update_button_states(self):
        """更新按钮启用状态"""
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.image_paths) - 1)

    def update_info(self):
        """更新信息显示"""
        current_image = os.path.basename(self.image_paths[self.current_index])
        self.info_label.setText(
            f"图片 {self.current_index + 1}/{len(self.image_paths)}: {current_image} | "
            f"已标记 {len(self.coordinates)} 个点"
        )

        # 更新坐标显示
        if self.coordinates:
            coord_text = "\n".join([
                f"点 {i + 1}: ({orig_x}, {orig_y})"
                for i, (orig_x, orig_y) in enumerate(self.coordinates)
            ])
            self.coords_edit.setText(coord_text)
        else:
            self.coords_edit.setText("尚未标记任何点 (请点击平面图中楼梯入口/出口确定楼梯节点位置)")

    def mousePressEvent(self, event):
        """处理鼠标点击事件，转换坐标"""
        # 只在图像标签上处理左键点击
        if event.button() == Qt.LeftButton:
            # 确保有原始图片和缩放后的图片
            if not self.orig_pixmap or not self.scaled_pixmap:
                return

            # 获取缩放比例
            orig_width = self.orig_pixmap.width()
            orig_height = self.orig_pixmap.height()
            scaled_width = self.scaled_pixmap.width()
            scaled_height = self.scaled_pixmap.height()

            # 计算图像在QLabel中的偏移（居中显示）
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            offset_x = (label_width - scaled_width) // 2
            offset_y = (label_height - scaled_height) // 2

            # 获取点击位置
            click_pos = event.pos() - self.image_label.pos()

            # 将点击位置转换为相对于缩放后图像的位置
            scaled_x = click_pos.x() - offset_x
            scaled_y = click_pos.y() - offset_y

            # 检查点击是否在图像范围内
            if 0 <= scaled_x < scaled_width and 0 <= scaled_y < scaled_height:
                # 计算原始图像坐标
                orig_x = int(scaled_x * orig_width / scaled_width)
                orig_y = int(scaled_y * orig_height / scaled_height)

                # 添加到坐标列表
                self.coordinates.append((orig_x, orig_y))

                # 在图像上绘制点 (红色十字)
                pixmap_copy = self.scaled_pixmap.copy()
                painter = QPainter(pixmap_copy)
                painter.setPen(QPen(Qt.red, 2))

                # 在缩放图像上绘制点，位置按比例映射
                draw_x = scaled_x
                draw_y = scaled_y
                size = min(scaled_width, scaled_height) * 0.015  # 十字大小自适应
                painter.drawLine(draw_x - size, draw_y, draw_x + size, draw_y)
                painter.drawLine(draw_x, draw_y - size, draw_x, draw_y + size)
                painter.end()

                # 更新显示的图像
                self.image_label.setPixmap(pixmap_copy)

                # 更新信息显示
                self.update_info()

    def prev_image(self):
        """切换到上一张图片"""
        if self.current_index > 0:
            # 保存当前坐标
            self.all_coordinates.append(self.coordinates)

            # 切换到上一张
            self.current_index -= 1
            self.coordinates = []
            self.load_image()

    def next_image(self):
        """切换到下一张图片"""
        if self.current_index < len(self.image_paths) - 1:
            # 保存当前坐标
            self.all_coordinates.append(self.coordinates)

            # 切换到下一张
            self.current_index += 1
            self.coordinates = []
            self.load_image()

    def clear_coordinates(self):
        """清除当前图像的坐标"""
        self.coordinates = []

        # 重新加载图像（不清除标记）
        self.image_label.setPixmap(self.scaled_pixmap)
        self.update_info()

    def save_and_exit(self):
        """保存所有坐标并关闭窗口"""
        # 添加当前图片的坐标
        if self.current_index < len(self.all_coordinates):
            self.all_coordinates[self.current_index] = self.coordinates
        else:
            self.all_coordinates.append(self.coordinates)

        try:
            with open(self.output_file, "w") as f:
                # 按要求的格式保存：每行对应一个楼层，点之间用逗号分隔
                for coords in self.all_coordinates:
                    if coords:
                        # 格式化为 (x1, y1), (x2, y2), ...
                        coord_str = ", ".join([f"({x}, {y})" for x, y in coords])
                        f.write(f"{coord_str}\n")
                    else:
                        # 如果某楼层没有坐标，写入空行
                        f.write("\n")

            QMessageBox.information(self, "成功", f"坐标已保存到:\n{self.output_file}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存坐标失败: {str(e)}")

    def closeEvent(self, event):
        """关闭窗口时的确认"""
        reply = QMessageBox.question(
            self, "关闭确认",
            "确定要关闭窗口吗？未保存的坐标将会丢失。",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )

        if reply == QMessageBox.Save:
            self.save_and_exit()
        elif reply == QMessageBox.Discard:
            event.accept()
        else:
            event.ignore()


def record_stairs(image_paths, output_file="stairs.txt"):
    """运行坐标标注工具"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # 创建对话框并运行
    dialog = CoordinateRecorderDialog(image_paths, output_file)
    result = dialog.exec_()

    return result == QDialog.Accepted