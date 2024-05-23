import random

import cv2
import numpy as np

import genetic_algorithm
import image_processing
import teeth_detection
from puzzle_extracting import ExtractedPuzzle, PuzzleCollection
from teeth_detection import get_vectors_from_corners, Vector


def create_puzzle(path):
    image = image_processing.load_image(path)
    puzzle = ExtractedPuzzle(image=image, mask=image_processing.threshold(image, 0))
    puzzle.find_corners()
    puzzle.find_notches()
    return puzzle


def find_matching_notches(puzzle1, puzzle2):
    notches = {}
    for type1, notch1 in puzzle1.notches.items():
        for type2, notch2 in puzzle2.notches.items():
            if notch1.does_match(notch2):
                notches[(type1, type2)] = (notch1, notch2)
    return notches


def _number_of_rotations2(type1, type2):  # for some reason this is not working??????
    types = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    distance = types.index(type2) + types.index(type1)
    return distance % 4


def number_of_rotations(type1: str, type2: str):
    # print("counting rotations:", type1, type2)
    types = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    a, b = types.index(type1), types.index(type2)
    # print(f"types: {a}, {b}, diff={a-b}")
    if a == b:  # if they are the same, do a 180
        return 2
    if (a - b) % 2 == 0:
        return 0
    return (4 + (a - b)) % 4


# def connect_puzzles(puzzle1, puzzle2):
#    notches = find_matching_notches(puzzle1, puzzle2)
#    for (type1, type2), (notch1, notch2) in notches.items():
#        print(type1, notch1," ", type2, notch2)
#        rotations = number_of_rotations(type1, type2)
#        rotated_puzzle2 = puzzle2.get_rotated(rotations)
#        print("viewing matching puzzles!!")
#        image_processing.view_image(puzzle1.get_preview(), type1)
#        image_processing.view_image(rotated_puzzle2.get_preview(), type2)

def get_opposite_edge(edge):
    types = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    index = types.index(edge)
    return types[(index + 2) % 4]


def place_image_in_image(background, image, point):
    if len(image.shape) == 2:
        shape = image.shape
        start_x, start_y = point
        background[start_y:start_y + shape[0], start_x:start_x + shape[1]] += image
    else:
        shape = image.shape
        start_x, start_y = point
        background[start_y:start_y + shape[0], start_x:start_x + shape[1], :] += image


def mask_puzzle_connection(image, a, padding=0.05) -> (np.ndarray, np.ndarray):
    """creates a mask for a puzzle connection, with a cross and a circle in the middle
    :param image: the image to be masked
    :param a: the size of the mask
    :param padding: crop the mask by this amount
    :returns: a tuple containing the following elements:
        mask (np.ndarray): The mask for the puzzle connection.
        cross_mask (np.ndarray): The cross mask for the puzzle connection."""
    a = int(a * (1 - padding))
    cropped, a = image_processing.crop_square(image, a)
    cross_mask = image_processing.get_cross(a, thickness=a * 0.2)
    # rhombus_mask = image_processing.get_rhombus(a)
    circle_mask = image_processing.get_circle(a, r=a // 3)
    mask = cv2.bitwise_or(circle_mask, cross_mask)
    # image_processing.view_image(mask)
    return cv2.bitwise_and(~cropped, mask), mask


def find_foremost_white_pixel(bounds, mask, y_start):
    start = bounds[0]
    end = bounds[1]
    if start[0] > end[0]:
        start, end = end, start

    loop = range(0)
    if y_start == "TOP":
        loop = range(0, mask.shape[0] // 2)
    if y_start == "BOTTOM":
        loop = range(mask.shape[0] - 1, mask.shape[0] // 2, -1)

    yList = []
    step = max(int(mask.shape[0] * 0.03), 3)
    for x in range(start[0], end[0], step):
        for y in loop:
            if mask[y][x] == 255:
                yList.append((x, y))
                break

    return yList


def avg_BGR(image, x, y):
    region = image[y:y + 3, x:x + 3]
    avg_rgb = np.mean(region, axis=(0, 1))

    return avg_rgb


def avg_puzzle_area(bottom_piece, i, puzzle1_white_pixel, puzzle2_white_pixel, top_piece):
    x1 = puzzle1_white_pixel[i][0]
    y1 = puzzle1_white_pixel[i][1]
    x2 = puzzle2_white_pixel[i][0]
    y2 = puzzle2_white_pixel[i][1]
    offset = int(bottom_piece.image.shape[0] * 0.03)
    avg_bgr_puzzle1 = avg_BGR(top_piece.image, x1, y1 - offset)
    avg_bgr_puzzle2 = avg_BGR(bottom_piece.image, x2, y2 + offset)
    return (avg_bgr_puzzle1, avg_bgr_puzzle2)


def count_distances(puzzle1_white_pixel, puzzle2_white_pixel, top_piece, bottom_piece):
    distance = []
    length = min(len(puzzle1_white_pixel), len(puzzle2_white_pixel))
    for i in range(length):
        avg_puzzle = avg_puzzle_area(bottom_piece, i, puzzle1_white_pixel, puzzle2_white_pixel, top_piece)
        distance.append(np.linalg.norm(np.array(avg_puzzle[0]) - np.array(avg_puzzle[1])))
    return distance


def scale_result(avg_distance):
    min = 15
    max = 150
    scaled_distance = np.clip((avg_distance - min) / (max - min), 0, 1)
    return scaled_distance


def calculate_image_similarity(puzzle1, edge_type1, puzzle2, edge_type2):
    n1 = number_of_rotations(edge_type1, "TOP")
    top_piece = puzzle1.get_rotated(n1)
    n2 = number_of_rotations(edge_type2, "BOTTOM")
    bottom_piece = puzzle2.get_rotated(n2)



    vector1 = get_vectors_from_corners(top_piece.corners)["BOTTOM"]

    puzzle1_white_pixel = find_foremost_white_pixel(bounds=(vector1.point1, vector1.point2),
                                                    mask=top_piece.mask, y_start="BOTTOM")

    vector2 = get_vectors_from_corners(bottom_piece.corners)["TOP"]
    puzzle2_white_pixel = find_foremost_white_pixel(bounds=(vector2.point1, vector2.point2),
                                                    mask=bottom_piece.mask, y_start="TOP")

    shape1, shape2 = puzzle1.mask.shape, puzzle2.mask.shape
    image_shape = shape1[0] + shape2[0], shape1[1] + shape2[1], 3  # RGB
    # print("shape", image_shape)


    distances = count_distances(puzzle1_white_pixel, puzzle2_white_pixel, top_piece, bottom_piece)
    avg_distance = np.mean(distances)

    scaled_distance = scale_result(avg_distance)

    # print(avg_distance, scaled_distance)

    background = np.zeros(image_shape, dtype=np.uint8)
    connection_point1 = vector1.get_middle()
    connection_point2 = vector2.get_middle()

    place_image_in_image(background, top_piece.image, (
        background.shape[1] // 2 - connection_point1[0], background.shape[0] // 2 - connection_point1[1]))
    place_image_in_image(background, bottom_piece.image, (
        background.shape[1] // 2 - connection_point2[0], background.shape[0] // 2 - connection_point2[1]))
    # image_processing.view_image(background, title=str(scaled_distance))

    return scaled_distance


def strip_colors(image):
    return image if len(image.shape) == 2 else image[:, :, 0]


def get_mask_xor_ratio(puzzle1, edge_type1, puzzle2, edge_type2):
    """:returns:
        tuple: A tuple containing the following elements:
            similarity (float): A measure of similarity between the two edges.
                This value indicates how similar the shapes of notches are, with 1.0 being identical.
            length_similarity (float): A measure of similarity based on the length of the edges.
                This value gives an idea of how closely the dimensions or sizes of the images match, with 1.0 being identical.
            output_img (np.ndarray): The processed output image after comparison or modification.
                This image displays connected pieces as form of visualization.
    """
    shape1, shape2 = puzzle1.mask.shape, puzzle2.mask.shape
    image_shape = shape1[0] + shape2[0], shape1[1] + shape2[1]
    output_img1 = np.zeros(image_shape, dtype=np.uint8)
    output_img2 = np.zeros(image_shape, dtype=np.uint8)
    vector1 = get_vectors_from_corners(puzzle1.corners)[edge_type1]
    vector2 = get_vectors_from_corners(puzzle2.corners)[edge_type2]
    connection_point1 = vector1.get_middle()
    connection_point2 = vector2.get_middle()
    puzzle1_mask = strip_colors(puzzle1.mask)
    puzzle2_mask = strip_colors(puzzle2.mask)
    # puzzle1_mask = image_processing.erode(puzzle1_mask, 3)
    # puzzle2_mask = image_processing.erode(puzzle2_mask, 3)
    # puzzle1_mask = puzzle1.mask
    # puzzle2_mask = puzzle2.mask
    place_image_in_image(output_img1, puzzle1_mask, (
        output_img1.shape[1] // 2 - connection_point1[0], output_img1.shape[0] // 2 - connection_point1[1]))
    place_image_in_image(output_img2, puzzle2_mask, (
        output_img2.shape[1] // 2 - connection_point2[0], output_img2.shape[0] // 2 - connection_point2[1]))

    xor_img = np.bitwise_xor(output_img1, output_img2)

    close_up, mask = mask_puzzle_connection(xor_img, a=min(vector1.distance(), vector2.distance()))
    # close_up = cv2.erode(close_up, np.ones((3, 3), np.uint8), iterations=1)
    similarity1 = 1 - np.count_nonzero(close_up) / np.count_nonzero(mask)

    # white_area = np.count_nonzero(puzzle1_mask) + np.count_nonzero(puzzle2_mask)
    # white_area_xor = np.count_nonzero(xor_img)
    # similarity2 = white_area_xor / white_area

    # and_img = np.bitwise_and(output_img1, output_img2)
    # white_area_and = np.count_nonzero(and_img)
    # similarity3 = 1 - (white_area_and / white_area)

    # print(f"similarity = {similarity}, length_similarity = {length_similarity}")
    # print(f"compare similarities: similarity1={similarity1:.4}, similarity2={similarity2:.4}, similarity3={similarity3:.4}")
    length_similarity = 1 - abs(vector1.distance() - vector2.distance()) / max(vector1.distance(), vector2.distance())

    image_similarity = calculate_image_similarity(puzzle1, edge_type1, puzzle2, edge_type2)

    return similarity1, length_similarity, image_similarity, xor_img, close_up


def flat_sides_match(puzzle1, edge1, puzzle2, edge2):
    # check if flat side is along the edge of pair of puzzles
    directions = ["TOP", "RIGHT", "BOTTOM", "LEFT"]

    check_for_flats1 = [direction for direction in directions if (direction not in [edge1, get_opposite_edge(edge1)])]
    check_for_flats2 = [direction for direction in directions if (direction not in [edge2, get_opposite_edge(edge2)])]
    # check_for_flats contain all edges except connection edge and its opposite edge, because they don't matter

    flats1 = [flat for flat in check_for_flats1 if puzzle1.get_notch(flat) == teeth_detection.NotchType.NONE]
    flats2 = [flat for flat in check_for_flats2 if puzzle2.get_notch(flat) == teeth_detection.NotchType.NONE]

    if len(flats1) != len(flats2):  # puzzle has a flat side that the other doesn't
        return False
    if len(flats1) == 0:  # no flat sides
        return True
    if len(flats1) == 2:  # weird edge case, both puzzles have 2 flat sides (one dimensional puzzle)
        return True

    # check if the flat sides are on the same side
    next_edge1 = teeth_detection.get_next_type(edge1)
    prev_edge2 = teeth_detection.get_previous_type(edge2)
    # _print_flat_side_info(edge1, edge2, next_edge1, next_edge2, puzzle1, puzzle2)
    if puzzle1.get_notch(next_edge1) == teeth_detection.NotchType.NONE:
        if puzzle2.get_notch(prev_edge2) == teeth_detection.NotchType.NONE:
            return True  # flats are on the same side!!

    prev_edge1 = teeth_detection.get_previous_type(edge1)
    next_edge2 = teeth_detection.get_next_type(edge2)
    if puzzle1.get_notch(prev_edge1) == teeth_detection.NotchType.NONE:
        if puzzle2.get_notch(next_edge2) == teeth_detection.NotchType.NONE:
            return True  # flats are on the same side!!
    return False


def _print_flat_side_info(edge1, edge2, next_edge1, next_edge2, puzzle1, puzzle2):
    print("puzzle 1 items:")
    for edge in get_vectors_from_corners(puzzle1.corners):
        print(f"\t{edge}: {puzzle1.get_notch(edge)}")
    print("puzzle 2 items:")
    for edge in get_vectors_from_corners(puzzle2.corners):
        print(f"\t{edge}: {puzzle2.get_notch(edge)}")
    print("connecting by edges:", edge1, edge2)
    print("these are the notches:", puzzle1.get_notch(edge1), puzzle2.get_notch(edge2))
    print("and here are next edges:", next_edge1, next_edge2)
    print("also their notches:", puzzle1.get_notch(next_edge1), puzzle2.get_notch(next_edge2))


def connect_puzzles(puzzle1: ExtractedPuzzle, edge1: str, puzzle2: ExtractedPuzzle, edge2: str) -> (float, np.ndarray):
    # if not is_connection_possible(puzzle1, edge1, puzzle2, edge2): #should be checked before calling this function
    #    raise MatchException("connection is impossible!!")
    rotations = number_of_rotations(edge1, edge2)
    # preview1 = puzzle1.get_preview()
    puzzle1 = puzzle1.get_rotated(rotations)
    # preview2 = puzzle1.get_preview()
    # image_processing.view_image(image_processing.images_to_image([preview1, preview2]), f"rotated puzzles {rotations}")
    edge1 = get_opposite_edge(edge2)
    return get_mask_xor_ratio(puzzle1, edge1, puzzle2, edge2)


class MatchException(Exception):
    pass


class NotchesDoNotMatch(MatchException):
    pass


class FlatSidesDoNotMatch(MatchException):
    pass


def is_connection_possible(puzzle1, edge1, puzzle2, edge2):
    """checks if the notches match and if the flat sides are correct"""

    notch1, notch2 = puzzle1.get_notch(edge1), puzzle2.get_notch(edge2)
    if not notch1.does_match(notch2):
        raise NotchesDoNotMatch(f"notches do not match... {notch1} {notch2}")
    if not flat_sides_match(puzzle1, edge1, puzzle2, edge2):
        raise FlatSidesDoNotMatch("flat sides do not match...")
    return True


def test_connect_puzzles(pieces, edges):
    puzzle1, puzzle2 = pieces
    edge1, edge2 = edges
    print("edges: ", edge1, edge2)

    try:
        is_connection_possible(puzzle1, edge1, puzzle2, edge2)
    except FlatSidesDoNotMatch as e:
        print(e)
        raise e
    except NotchesDoNotMatch as e:
        print(e)
        raise e

    similarity, length_similarity, img1, img2 = connect_puzzles(puzzle1, edge1, puzzle2, edge2)
    return similarity, length_similarity, [img1, img2], number_of_rotations(edge1, edge2)


def test_random_pairs():
    print("testing random pairs!!")
    edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    puzzle_collection = PuzzleCollection.unpickle()
    puzzle_collection, _ = puzzle_collection.partition_by_notch_type(teeth_detection.NotchType.NONE)
    while True:
        try:
            random_indexes = random.sample(range(len(puzzle_collection.pieces)), 2)
            puzzle1, puzzle2 = puzzle_collection.pieces[random_indexes[0]], puzzle_collection.pieces[random_indexes[1]]
            edge1, edge2 = random.choice(edges), random.choice(edges)

            test_pair(edge1, edge2, puzzle1, puzzle2, random_indexes)
            # view_all_ids(puzzle_collection)
        except MatchException as e:
            print(f"\tError: {e}")


def test_pair(edge1, edge2, puzzle1, puzzle2, indexes):
    # image_processing.view_image(puzzle1.get_preview(), edge1)
    # image_processing.view_image(puzzle2.get_preview(), edge2)
    similarity, length_similarity, imgs, rotate_value = test_connect_puzzles((puzzle1, puzzle2), (edge1, edge2))

    print(f"testing pairs {indexes[0]} and {indexes[1]}, {edge1} and {edge2}")
    image = image_processing.images_to_image([puzzle1.get_preview(), puzzle2.get_preview()])
    image_processing.view_image(image, f"pair {indexes[0]} and {indexes[1]}")

    print(
        f"similarity = {similarity}, length_similarity = {length_similarity}, indexes = ({indexes[0]}, {indexes[1]}, \"{edge1}\", \"{edge2}\"),")
    print(f"notches: {puzzle1.get_notch(edge1)} {puzzle2.get_notch(edge2)}, rotations_needed = {rotate_value}")
    print(genetic_algorithm.calculate_similarity(similarity, length_similarity))
    imgs.extend([puzzle1.get_preview(), puzzle2.get_preview()])
    result = image_processing.images_to_image(imgs)
    image_processing.view_image(result, similarity)


def test_pairs(pairs):
    print("testing pairs!!")
    for match in pairs:
        try:
            index1, index2, edge1, edge2 = match
            puzzle_collection = PuzzleCollection.unpickle()
            puzzle_collection, _ = puzzle_collection.partition_by_notch_type(teeth_detection.NotchType.NONE)
            puzzle1, puzzle2 = puzzle_collection.pieces[index1], puzzle_collection.pieces[index2]

            test_pair(edge1, edge2, puzzle1, puzzle2, (index1, index2))
        except ValueError as e:
            print(f"\tError: {e}")


def view_all_ids(puzzle_collection):
    print(len(puzzle_collection.pieces))
    for i, piece in enumerate(puzzle_collection.pieces):
        print(f"viewing piece {i} with id {piece.id}")
        image = image_processing.images_to_image([piece.get_preview(), piece.mask])
        image_processing.view_image(image, f"piece {i}")


if __name__ == '__main__':
    matching_pairs = [
        (28, 2, "TOP", "LEFT"),
        (6, 0, "TOP", "TOP"),
        (1, 20, "TOP", "RIGHT"),
        (22, 23, "TOP", "LEFT"),
        (27, 8, "BOTTOM", "RIGHT"),
        (8, 13, "TOP", "RIGHT"),
        (14, 1, "RIGHT", "BOTTOM"),
    ]
    problematic_pairs = [
        (14, 8, "BOTTOM", "RIGHT"),
    ]

    test = [
        (23, 1, "LEFT", "TOP"),  # should work
        (9, 0, "TOP", "RIGHT"),  # should not work
        (11, 29, "RIGHT", "BOTTOM"),  # should not work
        (22, 29, "RIGHT", "TOP"),  # should not work
    ]

    # test_pairs([(6, 0, "LEFT", "RIGHT")])
    # test_pairs(matching_pairs)
    # test_pairs(problematic_pairs)
    test_random_pairs()
