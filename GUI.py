import sys
import threading

from PyQt5.QtGui import QIcon
from qtpy.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFileDialog)
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt
from tqdm import tqdm

import image_processing
import puzzle_snake
from genetic_algorithm import Evolution, fitFun, apply_images_to_puzzles, save_snake
from photo_processing import process_photo
from puzzle_extracting import extract_puzzles
from teeth_detection import NotchType


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
        self.image_label.setWordWrap(True)
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
        self.button_layout.addWidget(self.btn_add_file)
        self.button_layout.addWidget(self.btn_solve_puzzle)

        # Add widgets to main layout
        self.layout.addWidget(self.image_label)
        self.layout.addLayout(self.button_layout)

        # Connect buttons to functions
        self.btn_add_file.clicked.connect(self.add_file)
        self.btn_solve_puzzle.clicked.connect(self.solve_puzzle_onclick)

        # Variable to hold the current file path
        self.current_file_path = None

    def add_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.current_file_path = file_name
            self.display_image(file_name)

    def solve_puzzle_onclick(self):
        # Display a loading message or screen
        if self.current_file_path is None:
            self.image_label.setText(f"select file by selecting image in \"Add file\" button")
        else:
            self.image_label.setText(f"Solving the puzzle {self.current_file_path}, please wait...")
            threading.Thread(target=self.solve_puzzle).start()

    def solve_puzzle(self):
        # name = "scattered_jaszczur_v=4_r=False"
        name = self.current_file_path
        path = name

        image, mask = process_photo(path)

        puzzle_collection = extract_puzzles(image, mask, rotate=True)

        big_preview = puzzle_collection.get_preview()
        image_processing.view_image(big_preview, title="log")

        genetic_solve(puzzle_collection)
        self.display_image("/results/guiResult.png")

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


def genetic_solve(puzzle_collection):
    puzzle_collection, _ = puzzle_collection.partition_by_notch_type(NotchType.NONE)
    puzzle_collection.set_ids()
    # image_processing.view_image(puzzle_collection.get_preview(),"edge pieces")
    edge_pieces = puzzle_collection.pieces

    num_of_iterations = 100
    num_of_chromosomes = 100
    num_of_genes = len(edge_pieces)
    desired_fit = 0.5

    evolution = Evolution(num_of_chromosomes, num_of_genes, 0, 0.1, 0.2, do_rotate=True)

    record_fit = num_of_genes * 2
    for it in tqdm(range(num_of_iterations)):
        evolution.iteration()

        best_chromosome = evolution.get_best_chromosome()
        best_fit, fitness_logs = fitFun(best_chromosome, get_fits=True)

        if it % 100 == 0:
            print(f" sum of fits: {evolution.get_sum_of_fits():.2f}", end=" ")
            print(f"best fit: {best_fit:.3f}", end=" ")
            print(f"piece ids: {[piece.id for piece in best_chromosome]}")

        if (best_fit < record_fit) or (it == num_of_iterations - 1) or (best_fit < desired_fit):
            record_fit = best_fit
            fitFun(best_chromosome, print_fits=False)
            print(f"best fit: {best_fit:.3f}")

    best_chromosome = evolution.get_best_chromosome()
    apply_images_to_puzzles(best_chromosome)
    snake_animation = puzzle_snake.get_snake_animation(best_chromosome, show_animation=False)
    print("saving")
    image_processing.save_image("./results/guiResult.png", snake_animation[-1])


def launch_application():
    # Create an instance of QApplication
    app = QApplication(sys.argv)
    # Create an instance of main window
    main_window = MainWindow()
    main_window.show()
    # Execute the application
    sys.exit(app.exec_())


if __name__ == '__main__':
    launch_application()
