from enum import Enum

import cv2
import numpy as np
import bresenham
import image_processing



class CornerType(Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3

def draw_circle(image, circle_coords, radius:int, color):
    x, y = circle_coords
    piece_with_hole = image.copy()
    cv2.circle(piece_with_hole, (x, y), radius, color, -1)
    return piece_with_hole



def get_image_slices(vector, image):
    slice1, slice2 = np.zeros_like(image), np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if vector.as_function()(j) > i:
                slice1[i][j] = [255,255,255]
            else:
                slice2[i][j] = [255,255,255]
    return slice1, slice2


class Vector:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2
    def as_function(self):
        x1, y1 = self.point1
        x2, y2 = self.point2

        if x1 == x2:
            x1 += 1

        a = (y1-y2)/(x1-x2)
        b = y1 - a*x1

        return lambda x: a*x + b

    def get_point_between(self):
        return (self.point1[0] + self.point2[0]) // 2, (self.point1[1] + self.point2[1]) // 2
    def __str__(self):
        return f"Vector: {self.point1} -> {self.point2}"


def move_towards(point1, center, percentage=0.1):
    x1, y1 = point1
    x2, y2 = center

    x = int(x1 + (x2-x1)*percentage)
    y = int(y1 + (y2-y1)*percentage)

    return x, y


def count_flips(pixels):
    flips = 0
    for i in range(len(pixels)-1):
        if pixels[i] != pixels[i+1]:
            flips += 1
    return flips


def convert_rgb_to_binary(pixel):
    if pixel[0] == 0:
        return 0
    return 255


def is_there_hole_inside(vector, image):
    """check if there is a hole inside the piece, by checking if there are 2 flips in the pixels on the vector line"""
    center = (image.shape[1]//2, image.shape[0]//2)
    point1 = move_towards(vector.point1, center, percentage=0.1)
    point2 = move_towards(vector.point2, center, percentage=0.1)

    coords = bresenham.connect(point1, point2)
    pixels = [convert_rgb_to_binary(image[coord[1],coord[0]]) for coord in coords]

    # preview = draw_circle(image, point1, 3, (0, 0, 255))
    # preview = draw_circle(preview, point2, 3, (0, 0, 255))
    # preview = cv2.line(preview, point1, point2, (0, 0, 255), 1)
    # print(pixels)
    # image_processing.view_image(preview)

    amount = count_flips(pixels)
    if amount == 2:
        if np.unique(pixels, return_counts=True)[1][0] <= 2: #TODO remove this hack. fix generator instead
            return False
        return True
    if amount == 0:
        return False
    raise Exception("There should be 2 flips or 0 flips")


def is_there_knob(knob_check, mask):
    ratio = np.count_nonzero(knob_check) / np.count_nonzero(mask)

    if ratio > 0.05:
        return True


class NotchType(Enum):
    NONE = 0,
    HOLE = 1,
    TOOTH = 2


def get_vectors_from_corners(corners):
    vectors = {
        "TOP": Vector(corners[0], corners[1]),
        "RIGHT": Vector(corners[1], corners[3]),
        "LEFT": Vector(corners[0], corners[2]),
        "BOTTOM": Vector(corners[2], corners[3])
    }
    return vectors


def get_teeth(puzzle_image, corners):


    vectors = get_vectors_from_corners(corners)

    edges_info = {}

    for type, vector in vectors.items():

        #outside knobs
        slices = get_image_slices(vector, puzzle_image)
        outside_mask = min(slices, key=lambda x: np.count_nonzero(x))
        #image_processing.view_image(outside_mask)
        knob_check = cv2.bitwise_and(puzzle_image, outside_mask)
        #image_processing.view_image(knob_check, title="knob_check")


        if is_there_hole_inside(vector, puzzle_image):
            result = NotchType.HOLE
        elif is_there_knob(knob_check, outside_mask):
            result = NotchType.TOOTH
        else:
            result = NotchType.NONE
        #image_processing.view_image(knob_check,title = ratio)
        edges_info[type] = result
    #image_processing.view_image(puzel)
    return edges_info



if __name__ == '__main__':
    puzzle_image = image_processing.load_image("testowy2.png")
    corners = [
        (43, 53),
        (165, 53),
        (43, 174),
        (165, 174)
    ]


    info = get_teeth(puzzle_image, corners)
    for type, result in info.items():
        print(type, result)
    print()
    image_processing.view_image(puzzle_image)





