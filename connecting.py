import random

import numpy as np

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


def number_of_rotations(type1:str, type2:str):
    #print("counting rotations:", type1, type2)
    types = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    a = types.index(type1)
    b = types.index(type2)
    #print(f"types: {a}, {b}, diff={a-b}")

    if a == b: #if they are the same, do a 180
        return 2
    if (a - b) % 2 == 0:
        return 0
    else:
        return (4+(a-b))%4


#def connect_puzzles(puzzle1, puzzle2):
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
    image = image if len(image.shape) == 2 else image[:, :, 0] #strip the 3rd dimension if it exists TODO remove this
    shape = image.shape
    start_x, start_y = point
    background[start_y:start_y + shape[0], start_x:start_x + shape[1]] += image


def get_mask_xor_ratio(puzzle1, edge_type1, puzzle2, edge_type2) -> (float, np.ndarray):
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
    place_image_in_image(output_img1, puzzle1.mask, (output_img1.shape[1] // 2 - connection_point1[0], output_img1.shape[0] // 2 - connection_point1[1]))
    place_image_in_image(output_img2, puzzle2.mask, (output_img2.shape[1] // 2 - connection_point2[0], output_img2.shape[0] // 2 - connection_point2[1]))
    white_area = np.count_nonzero(puzzle1.mask) + np.count_nonzero(puzzle2.mask)

    xor_img = np.bitwise_xor(output_img1, output_img2)
    white_area_xor = np.count_nonzero(xor_img)
    similarity = white_area_xor / white_area

    and_img = np.bitwise_and(output_img1, output_img2)
    #white_area_and = np.count_nonzero(and_img)
    #similarity = 1 - (white_area_and / white_area)

    length_similarity = 1 - abs(vector1.distance() - vector2.distance()) / max(vector1.distance(), vector2.distance())
    #print(f"similarity = {similarity}, length_similarity = {length_similarity}")
    return similarity, length_similarity, xor_img, and_img


def number_of_rotations2(type1, type2):
    types = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    distance = types.index(type2) + types.index(type1)
    return distance % 4



def flat_sides_match(puzzle1, edge1, puzzle2, edge2):
    #check if flat side is along the edge of pair of puzzles
    directions = ["TOP", "RIGHT", "BOTTOM", "LEFT"]

    check_for_flats1 = directions.copy()
    check_for_flats1.remove(edge1)
    check_for_flats1.remove(get_opposite_edge(edge1))
    #check_for_flats1 contains all edges except edge1 and its opposite edge

    check_for_flats2 = directions.copy()
    check_for_flats2.remove(edge2)
    check_for_flats2.remove(get_opposite_edge(edge2))
    #check_for_flats2 contains all edges except edge2 and its opposite edge

    flats1=[flat for flat in check_for_flats1 if puzzle1.notches[flat] == teeth_detection.NotchType.NONE]
    flats2=[flat for flat in check_for_flats2 if puzzle2.notches[flat] == teeth_detection.NotchType.NONE]
    if len(flats1) != len(flats2):
        #print("puzzle has a flat side that the other doesn't!!")
        return False
    if len(flats1) == 0: #if there are no flat sides, they match
        #print("no flats, match!!")
        return True
    rotations1 = number_of_rotations2(edge1, flats1[0])
    rotations2 = number_of_rotations2(edge2, flats2[0])
    if rotations1 != rotations2 and abs(rotations1 - rotations2) != 2:
        print(rotations1, rotations2)
        return False
    if len(flats1) == 2:
        rotations1 = number_of_rotations2(edge1, flats1[1])
        rotations2 = number_of_rotations2(edge2, flats2[1])
        if rotations1 != rotations2:
            #print("flats are on wrong sides!!")
            return False
   # print("flats match!!")
    return True


def connect_puzzles(puzzle1: ExtractedPuzzle, edge1: str, puzzle2: ExtractedPuzzle, edge2: str) -> (float, np.ndarray):

    #if not is_connection_possible(puzzle1, edge1, puzzle2, edge2): #should be checked before calling this function
    #    raise ValueError("connection is impossible!!")

    rotations = number_of_rotations(edge1, edge2)
    puzzle1 = puzzle1.deep_copy()
    puzzle1.rotate(rotations)
    edge1 = get_opposite_edge(edge2)
    #notch1 = puzzle1.get_notch(edge1)
    #image_processing.view_image(puzzle1.get_preview(), edge1)
    #if not notch1.does_match(notch2):
    #    raise ValueError(f"cannot connect {notch1} with {notch2}!! 2nd time!!")
    return get_mask_xor_ratio(puzzle1, edge1, puzzle2, edge2)


def is_connection_possible(puzzle1, edge1, puzzle2, edge2):
    """checks if the notches match and if the flat sides are correct"""
    notch1 = puzzle1.get_notch(edge1)
    notch2 = puzzle2.get_notch(edge2)
    if not notch1.does_match(notch2):
        #print(f"cannot connect {notch1} with {notch2}!!")
        return False
    if not flat_sides_match(puzzle1, edge1, puzzle2, edge2):
        #print(f"flat sides are wrong!!")
        return False
    return True


def test_connect_puzzles(pieces, edges):
    puzzle1, puzzle2 = pieces
    edge1, edge2 = edges
    print("edges: ", edge1, edge2)

    if not is_connection_possible(puzzle1, edge1, puzzle2, edge2):
        raise ValueError("connection is impossible!!")

    similarity, length_similarity, img1, img2 = connect_puzzles(puzzle1, edge1, puzzle2, edge2)
    masks = image_processing.images_to_image([img1, img2])
    pair = image_processing.images_to_image([puzzle.get_preview() for puzzle in [puzzle1, puzzle2]])
    return similarity, length_similarity, masks, pair


def test_random_pairs():
    print("testing random pairs!!")
    edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    puzzle_collection = PuzzleCollection.unpickle()
    while True:
        try:
            random_indexes = random.sample(range(len(puzzle_collection.pieces)), 2)
            puzzle1, puzzle2 = puzzle_collection.pieces[random_indexes[0]], puzzle_collection.pieces[random_indexes[1]]
            edge1, edge2 = random.choice(edges), random.choice(edges)
            similarity, length_similarity, mask, pair = test_connect_puzzles((puzzle1, puzzle2), (edge1, edge2))
            print(f"similarity = {similarity}, length_similarity = {length_similarity}, indexes = {random_indexes}, edges = {edge1, edge2}")
            image_processing.view_image(pair, "pair")
            image_processing.view_image(mask, similarity)
        except ValueError as e:
            print(f"\tError: {e}")

def test_pairs(pairs):
    print("testing pairs!!")
    for match in pairs:
        index1, index2, edge1, edge2 = match
        puzzle_collection = PuzzleCollection.unpickle()
        puzzle1, puzzle2 = puzzle_collection.pieces[index1], puzzle_collection.pieces[index2]
        try:
            similarity, length_similarity, mask, pair = test_connect_puzzles((puzzle1, puzzle2), (edge1, edge2))
            print(f"similarity = {similarity}, length_similarity = {length_similarity}")

            image_processing.view_image(pair, "pair")
            image_processing.view_image(mask, similarity)
        except ValueError as e:
            print(f"\tError: {e}")


if __name__ == '__main__':
    matching_pairs = [
        (28, 2, "TOP", "LEFT"),
        (6, 0, "TOP", "TOP"),
        (1, 20, "TOP", "RIGHT"),
    ]
    problematic_pairs = [
        (14, 8, "BOTTOM", "RIGHT"),
    ]

    #test_pairs(matching_pairs)
    #test_pairs(problematic_pairs)
    test_random_pairs()


