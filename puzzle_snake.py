import cv2

import genetic_algorithm
import image_processing
import numpy as np

from teeth_detection import NotchType
from teeth_detection import get_vectors_from_corners
from puzzle_extracting import PuzzleCollection, ExtractedPuzzle
from matching_puzzles import get_opposite_edge

def add_points(point1, point2):
    return point1[0] + point2[0], point1[1] + point2[1]
def subtract_points(point1, point2):
    return point1[0] - point2[0], point1[1] - point2[1]
def place_puzzle(background, puzzle, left_side, edge="LEFT"):
    edges = get_vectors_from_corners(puzzle.corners)
    connection_point = edges[edge].get_middle()

    image_start_pos = subtract_points(left_side, connection_point)

    background, image_start_pos = enlarge_background(background, image_start_pos, expand_amounts = puzzle.mask.shape[:2])

    end_y = image_start_pos[1] + puzzle.image.shape[0]
    end_x = image_start_pos[0] + puzzle.image.shape[1]

    if len(puzzle.mask.shape) == 2:
        puzzle.mask = cv2.cvtColor(puzzle.mask, cv2.COLOR_GRAY2BGR)
    background[image_start_pos[1]:end_y, image_start_pos[0]:end_x] = np.where(puzzle.mask > 0, puzzle.image, background[image_start_pos[1]:end_y, image_start_pos[0]:end_x])

    #cv2.circle(background, (int(image_start_pos[0]), int(image_start_pos[1])), 3, (0, 255, 0), -1)
    #cv2.circle(background, (int(left_side[0]), int(left_side[1])), 3, (0, 0, 255), -1)

    if NotchType.NONE not in puzzle.notches.values():
        next_edge = get_opposite_edge(edge)
    else:
        next_edge, _ = genetic_algorithm.edges_to_test(puzzle.notches)
    #print(next_edge)

    next_connection_point = edges[next_edge].get_middle()
    next_connection_point = add_points(next_connection_point, image_start_pos)
    return background, next_connection_point, next_edge


def enlarge_background(background, image_start_pos, expand_amounts):

    if image_start_pos[0] < 0 or image_start_pos[1] < 0:
        background, image_start_pos = expand_left_top(background, image_start_pos)

    new_width = max(background.shape[1], image_start_pos[0] + expand_amounts[1])
    new_height = max(background.shape[0], image_start_pos[1] + expand_amounts[0])

    if new_width != background.shape[1] or new_height != background.shape[0]:
        background = expand_right_bottom(background, new_height, new_width)

    return background, image_start_pos


def expand_right_bottom(background, new_height, new_width):

    new_background = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_background[:background.shape[0], :background.shape[1]] = background
    background = new_background
    return background


def expand_left_top(background, image_start_pos):

    shift_x = abs(min(0, image_start_pos[0]))
    shift_y = abs(min(0, image_start_pos[1]))

    new_width = background.shape[1] + shift_x
    new_height = background.shape[0] + shift_y
    new_background = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    new_background[shift_y:shift_y + background.shape[0], shift_x:shift_x + background.shape[1]] = background
    background = new_background

    image_start_pos = (image_start_pos[0] + shift_x, image_start_pos[1] + shift_y)
    return background, image_start_pos


def get_snake_image(puzzles, animation=False,show_image=False):
    """Place puzzles in a snake-like pattern. Used for edge pieces only!
    :param puzzles: list of ExtractedPuzzle objects
    :param animation: bool, if True, return list of images for animation
    :param show_image: bool, if True, show the image/animation
    :return: np.ndarray, image/animation with puzzles placed in a snake-like pattern"""

    width, height = 300, 300
    image = np.zeros((height, width, 3), dtype=np.uint8)

    connection_point = (100, 100)
    edge = "LEFT"
    animation =[]
    for i, puzzle in enumerate(puzzles):
        image, connection_point, next_edge = place_puzzle(image, puzzle, connection_point, edge)
        edge = get_opposite_edge(next_edge)
        if animation:
            animation.append(image.copy())
            if show_image:
                image_processing.view_image(image)
    if animation:
        return animation
    if show_image:
        image_processing.view_image(image)
    return image


def get_snake_animation(puzzles, show_animation=False):
    return get_snake_image(puzzles, animation=True, show_image=show_animation)


if __name__ == '__main__':
    collection = PuzzleCollection.unpickle("2024-04-26_bambi.pickle")
    pieces = collection.pieces
    #edge_pieces = collection.partition_by_notch_type(NotchType.NONE)[0].pieces

    while True:
        image = get_snake_image(pieces)
        image_processing.view_image(image)