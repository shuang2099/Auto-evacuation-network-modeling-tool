import math
import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QGroupBox, QScrollArea, QMessageBox, QGridLayout,
    QFrame, QSizePolicy, QDialog
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFont


class StaircaseCalculator(QDialog):
    calculationComplete = pyqtSignal()  # 添加完成信号

    def __init__(self, dir_path=None):
        super().__init__()
        self.setWindowTitle("楼梯参数计算器")
        self.setFixedSize(680, 600)  # 固定窗口尺寸，水平刚好显示内容

        # 设置应用字体
        app_font = QFont("Microsoft YaHei", 9)
        QApplication.setFont(app_font)

        # 存储所有楼梯参数输入框和结果
        self.stair_groups = []
        self.dir_path = ""
        self.folder_names = []

        # 创建主布局
        main_layout = QVBoxLayout(self)  # 直接设置主布局到对话框
        main_layout.setSpacing(5)  # 减少间距
        main_layout.setContentsMargins(10, 10, 10, 10)  # 减少边距

        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_content = QWidget()
        self.scroll_content_layout = QVBoxLayout(self.scroll_content)
        self.scroll_content_layout.setAlignment(Qt.AlignTop)
        self.scroll_content_layout.setSpacing(5)  # 减少楼梯组间距

        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #f8f8f8;")

        # 创建按钮
        self.calc_btn = QPushButton("计算并保存结果")
        self.calc_btn.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                color: #333333;
                border: 1px solid #b0b0b0;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover { background-color: #d0d0d0; }
            QPushButton:pressed { background-color: #c0c0c0; }
        """)
        self.calc_btn.setFixedHeight(40)
        self.calc_btn.clicked.connect(self.calculate_and_save)

        # 添加控件到主布局
        main_layout.addWidget(self.scroll_area, 1)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.calc_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # 设置目录
        if dir_path and os.path.isdir(dir_path):
            self.set_directory(dir_path)
        else:
            # 添加两个默认楼梯组
            for i in range(2):
                group = StairGroup(f"第{i + 1}层 到 第{i + 2}层", i)
                group.add_staircase()
                self.scroll_content_layout.addWidget(group)
                self.stair_groups.append(group)

    def set_directory(self, dir_path):
        """设置工作目录并更新UI"""
        if not os.path.isdir(dir_path):
            QMessageBox.warning(self, "路径错误", "指定的路径无效或不是文件夹")
            return

        try:
            self.dir_path = dir_path.replace('\\', '/')
            self.folder_names = [f for f in os.listdir(dir_path)
                                 if os.path.isdir(os.path.join(dir_path, f)) and f != 'Output.xlsx']
            self.folder_names.sort()
            self.update_ui_for_layers()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取文件夹时出错: {str(e)}")

    def update_ui_for_layers(self):
        """根据楼层数量更新楼梯组界面"""
        # 清除现有楼梯组
        for group in self.stair_groups:
            group.setParent(None)
        self.stair_groups = []

        # 创建新的楼梯组
        num_layers = len(self.folder_names)
        stair_segments = max(1, num_layers - 1)

        if num_layers >= 2:
            for i in range(stair_segments):
                group = StairGroup(f"第{i + 1}层 到 第{i + 2}层", i)
                group.add_staircase()
                self.scroll_content_layout.addWidget(group)
                self.stair_groups.append(group)
        else:
            group = StairGroup("第1层 到 第2层", 0)
            group.add_staircase()
            self.scroll_content_layout.addWidget(group)
            self.stair_groups.append(group)

    def get_stair_params(self):
        """获取所有楼梯参数"""
        all_params = []

        for group in self.stair_groups:
            group_params = []
            for stair_box in group.stair_boxes:
                # 验证并获取参数
                params = stair_box.get_params()
                if not params:
                    QMessageBox.warning(self, "输入错误", f"请填写'{group.title()}'的所有楼梯参数")
                    return None
                group_params.append(params)

            all_params.append(group_params)

        return all_params

    def calculate_and_save(self):
        """计算并保存结果"""
        # 验证文件夹
        if not self.dir_path or not os.path.isdir(self.dir_path):
            QMessageBox.warning(self, "路径错误", "请确保文件夹路径有效")
            return

        if len(self.folder_names) < 2:
            QMessageBox.warning(self, "文件夹错误", "选择的文件夹中至少需要有两层")
            return

        # 获取参数
        all_params = self.get_stair_params()
        if all_params is None:
            return

        # 计算楼梯长度和宽度
        stairs_length = []
        stairs_width = []

        for group_idx, group_params in enumerate(all_params):
            for param in group_params:
                # 计算楼梯长度
                step_length = math.sqrt(param['step_width'] ** 2 + param['step_height'] ** 2)
                total_length = step_length * param['step_count'] + param['platform_depth']
                total_length = total_length * 2 + param['width']  # 两侧楼梯+中间平台

                stairs_length.append(total_length)
                stairs_width.append(param['width'])

        # 保存结果
        save_dir = os.path.join(self.dir_path, self.folder_names[0])
        os.makedirs(save_dir, exist_ok=True)

        try:
            # 保存楼梯长度
            with open(os.path.join(save_dir, 'stairs_length.txt'), 'w') as f:
                for length in stairs_length:
                    f.write(f"{length:.4f}\n")

            # 保存楼梯宽度
            with open(os.path.join(save_dir, 'stairs_width.txt'), 'w') as f:
                for width in stairs_width:
                    f.write(f"{width:.4f}\n")

            QMessageBox.information(self, "成功", f"计算结果已保存到: {save_dir}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "保存错误", f"保存文件时出错: {str(e)}")


class StairGroup(QGroupBox):
    def __init__(self, title, group_idx):
        super().__init__(title)
        self.group_idx = group_idx
        self.stair_boxes = []

        # 设置组样式
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 8px;
                margin-bottom: 2px;
                font-weight: bold;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: padding;
                left: 10px;
                padding: 0 5px;
            }
        """)

        # 创建组布局
        self.group_layout = QVBoxLayout(self)
        self.group_layout.setSpacing(5)
        self.group_layout.setContentsMargins(10, 15, 10, 10)  # 减少边距

        # 添加楼梯按钮
        self.add_stair_btn = QPushButton("添加楼梯")
        self.add_stair_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #333333;
                border: 1px solid #b0b0b0;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #e0e0e0; }
            QPushButton:pressed { background-color: #d0d0d0; }
        """)
        self.add_stair_btn.clicked.connect(self.add_staircase)

        # 添加按钮到底部
        self.group_layout.addStretch()
        self.group_layout.addWidget(self.add_stair_btn, 0, Qt.AlignCenter)

        # 添加一个默认楼梯
        self.add_staircase()

    def add_staircase(self):
        """添加一个新的楼梯输入部分"""
        stair_idx = len(self.stair_boxes) + 1
        stair_box = StaircaseBox(stair_idx)
        self.stair_boxes.append(stair_box)
        self.group_layout.insertWidget(len(self.stair_boxes) - 1, stair_box)


class StaircaseBox(QGroupBox):
    def __init__(self, stair_id):
        super().__init__(f"楼梯{stair_id}")

        # 设置样式
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                margin-top: 4px;
                font-weight: normal;
            }
            QGroupBox::title {
                color: #555555;
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }
        """)

        # 使用网格布局确保精确对齐
        grid_layout = QGridLayout(self)
        grid_layout.setVerticalSpacing(8)
        grid_layout.setHorizontalSpacing(12)
        grid_layout.setContentsMargins(10, 18, 10, 10)  # 减少边距

        # 第一行参数
        # 踏步宽度
        self.step_width_edit = QLineEdit()
        self.step_width_edit.setFixedWidth(70)
        grid_layout.addWidget(QLabel("踏步宽度(m):"), 0, 0, Qt.AlignRight)
        grid_layout.addWidget(self.step_width_edit, 0, 1)

        # 踏步高度
        self.height_edit = QLineEdit()
        self.height_edit.setFixedWidth(70)
        grid_layout.addWidget(QLabel("踏步高度(m):"), 0, 2, Qt.AlignRight)
        grid_layout.addWidget(self.height_edit, 0, 3)

        # 踏步阶数
        self.count_edit = QLineEdit()
        self.count_edit.setFixedWidth(70)
        grid_layout.addWidget(QLabel("踏步阶数:"), 0, 4, Qt.AlignRight)
        grid_layout.addWidget(self.count_edit, 0, 5)

        # 第二行参数
        # 平台中心点宽度
        self.width_edit = QLineEdit()
        self.width_edit.setFixedWidth(70)
        grid_layout.addWidget(QLabel("楼梯宽度(m):"), 1, 0, Qt.AlignRight)
        grid_layout.addWidget(self.width_edit, 1, 1)

        # 平台深度（放在第二列）
        self.platform_edit = QLineEdit()
        self.platform_edit.setFixedWidth(70)
        grid_layout.addWidget(QLabel("平台深度(m):"), 1, 2, Qt.AlignRight)
        grid_layout.addWidget(self.platform_edit, 1, 3)

        # 设置列的最小宽度和拉伸因子
        grid_layout.setColumnMinimumWidth(0, 120)  # 标签列
        grid_layout.setColumnMinimumWidth(2, 95)
        grid_layout.setColumnMinimumWidth(4, 80)
        grid_layout.setColumnStretch(6, 1)  # 最后列拉伸填充空间

    def get_params(self):
        """获取当前楼梯参数并验证"""
        # 获取输入值
        width = self.width_edit.text().strip()
        platform_depth = self.platform_edit.text().strip()
        step_height = self.height_edit.text().strip()
        step_width = self.step_width_edit.text().strip()
        step_count = self.count_edit.text().strip()

        # 检查完整性
        if not all([width, platform_depth, step_height, step_width, step_count]):
            return None

        try:
            # 转换并验证参数
            params = {
                'width': float(width),
                'platform_depth': float(platform_depth),
                'step_height': float(step_height),
                'step_width': float(step_width),
                'step_count': int(step_count)
            }

            if any(val <= 0 for val in params.values() if isinstance(val, (int, float))):
                return None

            return params
        except ValueError:
            return None


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#
#     # 通过命令行参数获取文件夹路径
#     dir_path = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/GuYH/Desktop/test/temp"
#
#     window = StaircaseCalculator(dir_path)
#     window.show()
#     sys.exit(app.exec_())