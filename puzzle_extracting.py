import cv2
import numpy as np

import image_processing
import matplotlib.pyplot as plt

import test

import teeth_detection


def turn_into_binary(image):
    if len(image.shape) == 2:
        return image

    result = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] > 0:
                result[i][j] = 255
    return result


def detect_puzzles(mask):
    mask = turn_into_binary(mask)
    _, labels = cv2.connectedComponents(mask)
    puzzles = []
    for i in range(1, labels.max() + 1):
        puzzle = (labels == i).astype(np.uint8)
        puzzles.append(puzzle)
    return puzzles


def bound_puzzle(puzzle, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    bound = puzzle[y:y+h, x:x+w]
    return bound


def rotate(image, angle):
    #bound = bound_puzzle(puzzle,mask)
    bound = image_processing.enlarge_image(image, 1.5)
    height, width = bound.shape[:2]
    center = (width//2,height//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(bound,rotation_matrix, (width,height))
    return rotated_image


def bound_image(edges):

    x, y, w, h = cv2.boundingRect(edges)
    bound = edges[y:y + h, x:x + w]
    return bound, (x,y)


def num_of_edges(image):
    edges = cv2.Canny(image, 10, 200)
    #image_processing.view_image(edges)
    bounded, _ = bound_image(edges)
    edges = bounded//255


    num = 0
    for line in edges:
        #print("LEN:",len(line))
        counts = np.bincount(line)

        if len(counts) == 2:
            if counts[1] > 0.3 * counts[0]:
                # print(f"counts: {counts[0]}, {counts[1]}")
                num += 1
                #image_processing.view_image(edges)
    return num

def get_corners(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges = cv2.Canny(image * 255, 10, 200)
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    edges, point = bound_image(edges)
    edges = edges//255
    img = edges.copy()
    corners = []
    y = 0
    for line in edges:
        counts = np.bincount(line)

        if len(counts) == 2:
            if counts[1] > 0.3 * counts[0]:
                indexes = np.argwhere(line == 1)
                x1 = indexes[0][0]
                x2 = indexes[-1][0]
                corners.append(((x1, y), (x2, y)))
        y += 1

    max_field = 0
    best_corners = []

    for a1, a2 in corners:
        for b1, b2 in corners:
            field = min(abs(a1[0] - a2[0]), abs(b1[0] - b2[0])) * abs(a1[1] - b2[1])
            if field > max_field:
                max_field = field
                best_corners = [a1, a2, b1, b2]

    for corner in best_corners:
        cv2.circle(edges, (corner[0], corner[1]), 5, 255, -1)

    #image_processing.view_image(img * 255)
    #image_processing.view_image(edges)
    return [(corner[0] + point[0], corner[1] + point[1]) for corner in best_corners]


def get_puzzles_from_masks(image, masks):
    puzzles = []
    for mask in masks:
        puzzle = image * mask[:, :, None]

        puzzle = bound_puzzle(puzzle, mask)
        masked_puzzle = bound_puzzle(mask * 255, mask)

        puzzles.append((puzzle, masked_puzzle))
    return puzzles


""" 
    groups values, which values are close to each other 
    ex. [1,2,3,6,7,8,9] -> [[1,2,3],[6,7,8,9]]
"""
def find_neighbours(angles, distance=5):
    result = []

    for angle in angles:
        subset = [r for r in angles if abs(r - angle) < distance]
        if subset not in result:
            result.append(subset)
    return result

def find_rotation(puzzles):
    selected_puzzles = []
    puzzle_corners = []
    for i, (puzzle, mask) in enumerate(puzzles):
        #print(f"#{i}")
        # image_processing.view_image(puzzle)

        angles = []

        for angle in range(0, 360):

            rotated_puzzle = rotate(mask * 255, angle)
            #image_processing.view_image(rotated_puzzle*255)
            num = num_of_edges(rotated_puzzle * 255)
            if num > 1:
                angles.append(angle)
                #print(f"\tangle: {angle}, edges: {num}")
                # image_processing.view_image(rotated_puzzle*255)
        if len(angles) == 0:
            print("no edges found!!!!!!!!!!")
            exit(1)

        groups = find_neighbours(angles)
        largest_group = max(groups, key=len)
        median_element = np.median(largest_group)
        selected_puzzle = rotate(puzzle, median_element)
        corners = get_corners(rotate(mask * 255, median_element))

        selected_puzzles.append(selected_puzzle)
        puzzle_corners.append(corners)
    return selected_puzzles, puzzle_corners


def extract_puzzles(path):
    #wygenerowane puzzle
    image = image_processing.load_image(path)
    mask = image_processing.load_image(path.replace(".", "_mask."))


    masks = detect_puzzles(mask)
    print(f"number of puzzles: {len(masks)}")
    selected_puzzles = get_puzzles_from_masks(image, masks)
    rotated_puzzles, corners = find_rotation(selected_puzzles)
    return rotated_puzzles, corners






if __name__ == '__main__':
    path = "results/processed_photo.png"
    #path = "results/generated.png"
    puzzles, corners = extract_puzzles(path)


    for i, (puzzle, corners) in enumerate(zip(puzzles, corners)):
        info = teeth_detection.get_teeth(puzzle, corners)
        for type, result in info.items():
            print(type, result)
        image_processing.view_image(puzzle, title=f"puzzle_{i}")

        #image_processing.save_image(f"extracted/puzzle_{i}.png", puzzle)

