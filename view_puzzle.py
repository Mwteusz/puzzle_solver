import cv2

import genetic_algorithm
import image_processing
import numpy as np

import teeth_detection
from teeth_detection import NotchType
from puzzle_extracting import PuzzleCollection
from connecting import get_opposite_edge


def paste_image_on_image(background, puzzle, left_side, edge="LEFT"):
    print(left_side)
    edges = teeth_detection.get_vectors_from_corners(puzzle.corners)
    middle_left = edges[edge].get_middle()

    image_start_pos = (left_side[0] - middle_left[0], left_side[1] - middle_left[1])
    background[image_start_pos[1]:image_start_pos[1]+puzzle.image.shape[0], image_start_pos[0]:image_start_pos[0]+puzzle.image.shape[1]] += puzzle.image
    cv2.circle(background, image_start_pos, 3, (0, 255, 0), -1)
    cv2.circle(background, left_side, 3, (0, 0, 255), -1)

    if NotchType.NONE not in puzzle.notches.values():
        next_edge = get_opposite_edge(edge)
    else:
        next_edge, _ = genetic_algorithm.edges_to_test(puzzle.notches)
    print(next_edge)

    next_connection_point = edges[next_edge].get_middle()
    next_connection_point = (next_connection_point[0] + image_start_pos[0], next_connection_point[1] + image_start_pos[1])
    return next_connection_point, next_edge



def viewAllPuzzles(puzzles):
    #shuffling the puzzles
    np.random.shuffle(puzzles)
    width, height = 1000, 700
    image = np.zeros((height, width, 3),dtype=np.uint8)

    connection_point = (width//2, height//2)
    edge = "LEFT"
    for i, puzzle in enumerate(puzzles):

        connection_point, next_edge = paste_image_on_image(image, puzzle, connection_point,edge)
        edge = get_opposite_edge(next_edge)
        image_processing.view_image(image, "all puzzles")
    image_processing.view_image(image, "all puzzles")


if __name__ == '__main__':
    collection = PuzzleCollection.unpickle()
    puzzles = collection.pieces
    viewAllPuzzles(puzzles)
