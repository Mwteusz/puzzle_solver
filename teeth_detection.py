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


def view_line(perpendicular_line, shape, title=None):
    image = np.zeros(shape)
    for y in range(shape[0]):
        for x in range(shape[1]):
            if perpendicular_line.function(x) > y:
                image[y][x] = [255,255,255]
    image_processing.view_image(image, title=title)


def find_intersection(line, perpendicular_line):
    x = (perpendicular_line.b - line.b) / (line.a - perpendicular_line.a)
    y = line.a * x + line.b
    return x, y


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
    center = (image.shape[1]//2, image.shape[0]//2)
    point1 = move_towards(vector.point1, center, percentage=0.1)
    point2 = move_towards(vector.point2, center, percentage=0.1)

    preview = draw_circle(image, point1, 5, (0,0,255))
    preview = draw_circle(preview, point2, 5, (0,0,255))
    image_processing.view_image(preview)

    coords = bresenham.connect(point1, point2)
    pixels = [convert_rgb_to_binary(image[coord[1],coord[0]]) for coord in coords]



    amount = count_flips(pixels)
    if amount == 2:
        return True
    elif amount == 0:
        return False
    raise Exception("There should be 2 flips or 0 flips")


def is_there_knob(knob_check, mask):
    ratio = np.count_nonzero(knob_check) / np.count_nonzero(mask)

    if ratio > 0.05:
        return True


"""puzel to maska!!!"""
def get_teeth(puzel, corners):


    vectors = {}
    vectors["TOP"] = Vector(corners[0], corners[1])
    vectors["RIGHT"] = Vector(corners[1], corners[3])
    vectors["LEFT"] = Vector(corners[0], corners[2])
    vectors["BOTTOM"] = Vector(corners[2], corners[3])

    edges_info = {}

    for type, vector in vectors.items():

        #outside knobs
        slices = get_image_slices(vector, puzel)
        outside_mask = min(slices, key=lambda x: np.count_nonzero(x))
        #image_processing.view_image(outside_mask)
        knob_check = cv2.bitwise_and(puzel, outside_mask)
        #image_processing.view_image(knob_check, title="knob_check")


        if is_there_hole_inside(vector, puzel):
            result = "hole"
        elif is_there_knob(knob_check, outside_mask):
            result = "tooth"
        else:
            result = "none"
        #image_processing.view_image(knob_check,title = ratio)
        edges_info[type] = result
    #image_processing.view_image(puzel)
    return edges_info


if __name__ == '__main__':
    puzel = image_processing.load_image("testowy2.png")
    corners = [
        (43, 53),
        (165, 53),
        (43, 174),
        (165, 174)
    ]


    info = get_teeth(puzel, corners)
    for type, result in info.items():
        print(type, result)





