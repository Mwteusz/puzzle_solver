import sys

from PyQt5.QtGui import QIcon
from qtpy.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFileDialog)
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Main window setup
        self.setWindowTitle("Puzzle solver")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height
        self.setStyleSheet("background-color: #1E1E1E")
        self.setWindowIcon(QIcon('./logo/logo.ico'))

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main Layout
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Image label setup
        self.image_label = QLabel("Image will be showed here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(600, 400)
        self.image_label.setAcceptDrops(True)
        self.layout.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "font-size: 13px; border: 2px solid #343434; background-color: #2C2C2C; color: #DDDDDD")

        # Horizontal layout for buttons
        self.button_layout = QHBoxLayout()

        # Buttons
        self.btn_add_file = QPushButton("Add File")
        self.btn_solve_puzzle = QPushButton("Solve Puzzle")
        self.style_buttons()

        # Adding buttons to the horizontal layout
        self.button_layout.addWidget(self.btn_add_file, )
        self.button_layout.addWidget(self.btn_solve_puzzle)

        # Add widgets to main layout
        self.layout.addWidget(self.image_label)
        self.layout.addLayout(self.button_layout)

        # Connect buttons to functions
        self.btn_add_file.clicked.connect(self.add_file)
        self.btn_solve_puzzle.clicked.connect(self.solve_puzzle)

        # Variable to hold the current file path
        self.current_file_path = None

    def add_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        self.current_file_path = file_name
        if file_name:
            self.display_image(file_name)

    def solve_puzzle(self):
        # Display a loading message or screen
        self.image_label.setText(f"Solving the puzzle {self.current_file_path}, please wait...")

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage():
            event.accept()
        elif event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            self.display_image(file_path)

    def style_buttons(self):
        # Button styling
        button_style = """
        QPushButton {
            background-color: #333333;
            color: white;
            font-size: 16px;
            border: 2px solid #444444;
            padding: 5px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #555555;
        }
        QPushButton:pressed {
            background-color: #222222;
        }
        """
        self.btn_add_file.setStyleSheet(button_style)
        self.btn_solve_puzzle.setStyleSheet(button_style)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./logo/logo.ico'))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
