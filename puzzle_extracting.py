import cv2
import numpy as np

import image_processing
import matplotlib.pyplot as plt

import test


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
    return bound


def num_of_edges(image):
    edges = cv2.Canny(image, 10, 200)
    edges = bound_image(edges)//255


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
    for i, (puzzle, mask) in enumerate(puzzles):
        print(f"#{i}")
        # image_processing.view_image(puzzle)

        angles = []

        for angle in range(0, 360):

            rotated_puzzle = rotate(mask * 255, angle)
            #image_processing.view_image(rotated_puzzle*255)
            num = num_of_edges(rotated_puzzle * 255)
            if num > 1:
                angles.append(angle)
                print(f"\tangle: {angle}, edges: {num}")
                # image_processing.view_image(rotated_puzzle*255)
        if len(angles) == 0:
            print("no edges found!!!!!!!!!!")
            exit(1)

        groups = find_neighbours(angles)
        largest_group = max(groups, key=len)
        median_element = np.median(largest_group)
        selected_puzzle = rotate(puzzle, median_element)
        selected_puzzles.append(selected_puzzle)
    return selected_puzzles


def extract_puzzles(path):
    #wygenerowane puzzle
    image = image_processing.load_image(path)
    mask = image_processing.load_image(path.replace(".", "_mask."))


    masks = detect_puzzles(mask)
    print(f"number of puzzles: {len(masks)}")
    selected_puzzles = get_puzzles_from_masks(image, masks)
    selected_puzzles = find_rotation(selected_puzzles)
    return selected_puzzles






if __name__ == '__main__':
    path = "results/processed_photo.png"
    #path = "results/generated.png"
    extracted_puzzles = extract_puzzles(path)

    for i, puzzle in enumerate(extracted_puzzles):
        image_processing.save_image(f"extracted/puzzle_{i}.png", puzzle)

