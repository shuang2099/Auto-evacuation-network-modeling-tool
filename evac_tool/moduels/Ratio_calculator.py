import sys
import os
import math
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDialog, QLabel, QFileDialog, QTextEdit, QVBoxLayout, QHBoxLayout,
    QPushButton, QDesktopWidget, QInputDialog, QMessageBox
)


class CalibrationWindow(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像标定工具")
        self.setGeometry(100, 100, 800, 700)

        self.image_paths = image_paths  # 图片路径列表
        self.current_idx = 0  # 当前显示图片的索引
        self.scale_factor = 1.0
        self.coordinates = []  # 当前图片的坐标点
        self.factors = {}  # 存储每张图片的比例因子 {图片路径: 比例因子}

        # 创建UI组件
        self.create_ui()

        # 加载并显示第一张图片
        if self.image_paths:
            self.load_image(self.current_idx)
        else:
            QMessageBox.warning(self, "警告", "未找到任何可用图片")
            self.close()

    def create_ui(self):
        """创建用户界面布局"""
        # 主布局
        main_layout = QVBoxLayout()

        # 顶部导航按钮
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.prev_btn.setEnabled(False)  # 初始禁用

        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.show_next_image)

        self.copy_ratio_btn = QPushButton("与上一张比例相同")
        self.copy_ratio_btn.clicked.connect(self.copy_previous_ratio)
        # 修复1: 设置按钮初始启用状态为False
        self.copy_ratio_btn.setEnabled(False)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.copy_ratio_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn)

        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setCursor(Qt.CrossCursor)
        self.image_label.setMinimumHeight(400)

        # 状态显示
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)

        # 坐标显示区域
        self.coordinates_edit = QTextEdit()
        self.coordinates_edit.setReadOnly(True)
        self.coordinates_edit.setMaximumHeight(100)

        # 布局组装
        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.coordinates_edit)

        self.setLayout(main_layout)

    def load_image(self, idx):
        """加载并显示指定索引的图片"""
        if idx < 0 or idx >= len(self.image_paths):
            return

        self.current_idx = idx
        image_path = self.image_paths[idx]

        # 更新导航按钮状态
        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < len(self.image_paths) - 1)
        # 修复2: 只有在当前不是第一张图片时启用"与上一张相同"按钮
        self.copy_ratio_btn.setEnabled(idx > 0)

        # 清空当前坐标
        self.coordinates = []
        self.update_coordinates_display()

        # 更新状态显示
        self.status_label.setText(f"当前图片: {os.path.basename(image_path)} ({idx + 1}/{len(self.image_paths)})")

        # 加载图片
        self.original_image = QImage(image_path)
        if self.original_image.isNull():
            QMessageBox.warning(self, "错误", f"无法加载图片: {image_path}")
            return False

        # 缩放图片到适合窗口大小
        max_width = self.width() - 50
        max_height = self.height() - 200

        self.scale_factor = min(
            max_width / self.original_image.width(),
            max_height / self.original_image.height()
        )

        scaled_image = self.original_image.scaled(
            int(self.original_image.width() * self.scale_factor),
            int(self.original_image.height() * self.scale_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(QPixmap.fromImage(scaled_image))

        # 检查当前图片是否已有比例因子
        if image_path in self.factors:
            factor = self.factors[image_path]
            self.coordinates_edit.append(f"已有比例因子: {factor:.6f} 米/像素")

        return True

    def get_image_directory(self):
        """获取当前图片所在目录"""
        if self.current_idx < len(self.image_paths):
            return os.path.dirname(self.image_paths[self.current_idx])
        return ""

    def save_ratio(self, ratio):
        """保存比例因子到当前图片的目录"""
        if not self.image_paths:
            return False

        image_path = self.image_paths[self.current_idx]
        self.factors[image_path] = ratio

        # 保存到文件
        dir_path = self.get_image_directory()
        if dir_path:
            save_path = os.path.join(dir_path, "factor.txt")
            try:
                with open(save_path, "w") as f:
                    f.write(f"{ratio:.6f}")
                return True
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"无法保存比例文件: {str(e)}")
        return False

    def mousePressEvent(self, event):
        """处理鼠标点击事件"""
        if event.button() == Qt.LeftButton and hasattr(self, 'original_image'):
            # 将坐标转换为原始图像坐标
            pos = event.pos()
            pixmap = self.image_label.pixmap()
            if not pixmap:
                return

            # 计算相对于图像的位置
            img_x = (pos.x() - (self.image_label.width() - pixmap.width()) / 2)
            img_y = (pos.y() - (self.image_label.height() - pixmap.height()) / 2)

            # 转换为原始坐标
            raw_x = img_x / self.scale_factor
            raw_y = img_y / self.scale_factor

            # 记录坐标点
            if 0 <= raw_x < self.original_image.width() and 0 <= raw_y < self.original_image.height():
                self.coordinates.append((raw_x, raw_y))
                self.update_coordinates_display()

                # 有两点时计算比例
                if len(self.coordinates) == 2:
                    self.calculate_ratio()

    def update_coordinates_display(self):
        """更新坐标显示"""
        display_text = "已选点坐标：\n"
        for i, (x, y) in enumerate(self.coordinates):
            display_text += f"点 {i + 1}: ({x:.2f}, {y:.2f})\n"
        self.coordinates_edit.setText(display_text)

    def calculate_ratio(self):
        """计算实际比例"""
        try:
            # 获取两点距离
            if len(self.coordinates) < 2:
                return

            p1, p2 = self.coordinates[:2]
            distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            # 获取用户输入的实际距离
            actual_length, ok = QInputDialog.getDouble(
                self,
                "输入实际距离",
                "请输入两点间的实际距离（单位：米）:",
                min=0.01, max=10000.0, decimals=2
            )

            if ok and actual_length > 0:
                # 计算比例因子
                ratio = actual_length / distance

                # 保存结果
                if self.save_ratio(ratio):
                    self.coordinates_edit.append(f"\n计算比例因子: {ratio:.6f} 米/像素")
                    QMessageBox.information(self, "计算完成", f"比例因子已保存\n像素比：{ratio:.6f} 米/像素")
                else:
                    self.coordinates_edit.append(f"\n比例因子: {ratio:.6f} 米/像素 (未保存)")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算比例时出错: {str(e)}")
            self.coordinates = []

    def copy_previous_ratio(self):
        """复制上一张图片的比例因子"""
        if self.current_idx == 0:
            QMessageBox.warning(self, "警告", "这是第一张图片，没有上一张")
            return

        prev_image_path = self.image_paths[self.current_idx - 1]

        if prev_image_path in self.factors:
            prev_ratio = self.factors[prev_image_path]

            # 复制比例因子到当前图片
            if self.save_ratio(prev_ratio):
                self.coordinates_edit.append(f"\n已复制上一张比例: {prev_ratio:.6f} 米/像素")
                QMessageBox.information(self, "操作完成", f"已使用上一张图片的比例因子: {prev_ratio:.6f}")
            else:
                self.coordinates_edit.append(f"\n复制比例失败 (保存文件失败)")
        else:
            # 修复3: 尝试从文件中加载上一张的比例因子
            prev_dir = os.path.dirname(prev_image_path)
            factor_file = os.path.join(prev_dir, "factor.txt")

            if os.path.exists(factor_file):
                try:
                    with open(factor_file, "r") as f:
                        prev_ratio = float(f.read().strip())
                        # 保存到当前图片
                        if self.save_ratio(prev_ratio):
                            self.coordinates_edit.append(f"\n已复制上一张比例: {prev_ratio:.6f} 米/像素")
                            QMessageBox.information(self, "操作完成", f"已使用上一张图片的比例因子: {prev_ratio:.6f}")
                        else:
                            self.coordinates_edit.append(f"\n复制比例失败 (保存文件失败)")
                except Exception as e:
                    QMessageBox.warning(self, "警告", f"读取上一张比例因子失败: {str(e)}")
            else:
                QMessageBox.warning(self, "警告", "上一张图片尚未计算比例，也未找到factor.txt文件")

    def show_prev_image(self):
        """显示上一张图片"""
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)

    def show_next_image(self):
        """显示下一张图片"""
        if self.current_idx < len(self.image_paths) - 1:
            self.load_image(self.current_idx + 1)

    def resizeEvent(self, event):
        """处理窗口大小变化事件，重新调整图片大小"""
        super().resizeEvent(event)
        if hasattr(self, 'original_image') and not self.original_image.isNull():
            # 重新计算缩放比例
            max_width = self.width() - 50
            max_height = self.height() - 200

            self.scale_factor = min(
                max_width / self.original_image.width(),
                max_height / self.original_image.height()
            )

            scaled_image = self.original_image.scaled(
                int(self.original_image.width() * self.scale_factor),
                int(self.original_image.height() * self.scale_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_label.setPixmap(QPixmap.fromImage(scaled_image))