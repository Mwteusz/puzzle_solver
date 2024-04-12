import random

import cv2
import numpy as np


def is_coliding(puzzle, image, minimum_distance=1):
    #apply threshold
    binary_puzzle = cv2.threshold(puzzle, 0, 255, cv2.THRESH_BINARY)[1]
    binary_puzzle = cv2.dilate(binary_puzzle, np.ones((minimum_distance*2 + 1, minimum_distance*2 + 1), np.uint8), iterations=1)

    return np.any(binary_puzzle * image)



def scatter_pieces(size, pieces, minimum_distance=1):
    print("scattering pieces")
    grid_size_x, grid_size_y = size
    result = np.zeros((grid_size_x, grid_size_y, 3), dtype=np.uint8)
    random.shuffle(pieces)

    for i, piece in enumerate(pieces):
        center = (piece.puzzle_image.shape[0]//2, piece.puzzle_image.shape[1]//2)
        angle = random.randint(0, 359)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv2.warpAffine(piece.puzzle_image, rotation_matrix, piece.puzzle_image.shape[:2])
        puzzle_on_background = np.zeros_like(result)

        random_position = None
        while random_position is None or is_coliding(puzzle_on_background, result, minimum_distance):
            puzzle_on_background = np.zeros_like(result)
            random_position = (random.randint(0, result.shape[1] - rotated_image.shape[1]),
                               random.randint(0, result.shape[0] - rotated_image.shape[0]))
            puzzle_on_background[random_position[1]:random_position[1] + rotated_image.shape[0],    random_position[0]:random_position[0] + rotated_image.shape[1]] = rotated_image

        result += puzzle_on_background
        print(f"scattered piece {i+1}/{len(pieces)}")

    return result