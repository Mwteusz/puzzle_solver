import os
from datetime import date
import pickle
import timer
import cv2
import numpy as np
import image_processing
import teeth_detection
from progress_bar import ProgressBar
from teeth_detection import NotchType
from collections import deque as Deque
from functools import reduce

def view_corners(image, corners):
    preview = image.copy()
    for corner in corners:
        cv2.circle(preview, corner, 5, (255, 0, 0), -1)
    image_processing.view_image(preview, title="corners")


def rotate_point_90(point, rotations):
    rotations = rotations % 4
    rotations = 4 - rotations
    x, y = point
    for i in range(rotations):
        x, y = y, -x
    return x, y


def rotate_corners(corners, rotations, shape):
    rotations = rotations % 4
    corners = [rotate_point_90(corner, rotations) for corner in corners]
    v = (0, 0)
    if rotations == 3:
        v = (0, shape[0])
    elif rotations == 2:
        v = (shape[1], shape[0])
    elif rotations == 1:
        v = (shape[1], 0)

    corners = [(corner[0] + v[0], corner[1] + v[1]) for corner in corners]
    corners_queue = Deque(corners)
    corners_queue.rotate(rotations)
    corners = list(corners_queue)
    #print("corners after rotation", corners)
    return corners


def shift_notches(notches, rotations):
    rotations = rotations % 4
    rotations = rotations % 4
    new_notches = {}
    for type, notch in notches.items():
        new_type = type
        for i in range(rotations):
            new_type = teeth_detection.get_next_type(new_type)
        new_notches[new_type] = notch
    return new_notches


class ExtractedPuzzle:
    def __init__(self, notches: [NotchType] = None, image = None, mask = None, corners = None, id=None):
        self.notches = notches
        self.image = image
        self.mask = mask
        self.corners = corners
        self.id = id


    def __str__(self):
        str = ""
        for type, result in self.notches.items():
            str += f"{type}\t {result}\n"
        return str

    def rotate(self, rotations):
        """ rotations means 90 degree rotations, 1 rotation = 90 degrees, 2 rotations = 180 degrees, etc. """
        #print("rotating by", rotations)
        rotations = rotations % 4
        if self.image is not None:
            self.image = rotate_90(self.image, rotations)
        self.mask = rotate_90(self.mask, rotations)
        self.corners = rotate_corners(self.corners, rotations, self.mask.shape[:2])
        self.notches = shift_notches(self.notches, rotations)

    def deep_copy(self, copy_image=True):
        """returns a deep copy of the object, if copy_image is True, the image is copied as well, otherwise it is None"""
        new_corners = [(corner[0], corner[1]) for corner in self.corners]
        image_copy = None if (copy_image is False or self.image is None) else self.image.copy()
        return ExtractedPuzzle(notches=self.notches.copy(), image=image_copy, mask=self.mask.copy(), corners=new_corners, id=self.id)


    def find_corners(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.Canny(self.mask, 10, 200)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
        edges, point = image_processing.bound_image(edges)
        edges = edges // 255
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


        #test_corners = []
        for a1, a2 in corners:
            #test_corners.append((a1[0] + point[0], a1[1] + point[1]))
            #test_corners.append((a2[0] + point[0], a2[1] + point[1]))
            for b1, b2 in corners:
                field = min(abs(a1[0] - a2[0]), abs(b1[0] - b2[0])) * abs(a1[1] - b2[1])
                if field > max_field:
                    max_field = field
                    best_corners = [a1, a2, b2, b1]

        self.corners = [(corner[0] + point[0], corner[1] + point[1]) for corner in best_corners]

        #view_corners(self.image, test_corners)
        #image_processing.view_image(edges, title="edges")
        if len(self.corners) != 4:
            raise Exception(f"4 corners should be found, but found: {len(self.corners)}")

    def align_to_grid(self):
        mask = self.mask
        w, h = mask.shape
        if w > 100 or h > 100: #optimization, may not be needed
            mask = cv2.resize(self.mask, (100, 100))
        enlarged = image_processing.enlarge_image(mask, 1.5) #cache enlarged image
        angles = []
        for angle in range(0, 360):
            rotated_mask = rotate(enlarged, angle, enlarge=False) #enlarge=False,to prevent enlarging same image multiple times
            num = num_of_edges(rotated_mask)
            if num > 1: #TODO >  or >=
                angles.append(angle)
        if len(angles) == 0:
            raise Exception("No edges found!!!!!!!")

        groups = find_neighbours(angles)
        largest_group = max(groups, key=len)
        median_element = np.median(largest_group)

        self.mask = rotate(self.mask, median_element)
        self.mask = turn_into_binary(self.mask, 0.5)  # removes aliasing
        self.image = rotate(self.image, median_element)
        self.corners = None  # corners need to be recalculated
        return
    def find_notches(self):
        self.notches = teeth_detection.get_teeth(self.image, self.corners)
    def get_notch(self, type):
        return self.notches[type]

    def get_preview(self):
        if self.corners is None or self.notches is None:
            raise Exception("corners or notches not found")

        image = self.image.copy()
        for corner in self.corners:
            cv2.circle(image, corner, 5, (255,0,0), -1)

        for type, vector in teeth_detection.get_vectors_from_corners(self.corners).items():
            cv2.line(image, vector.point1, vector.point2, (255,128,64), 1)
            notch_type = self.notches[type]
            if notch_type is not NotchType.NONE:
                name = notch_type.name.removeprefix("NotchType.").capitalize()
                image_processing.put_text(image, name, vector.get_middle(), (0, 0, 0), 2)
                image_processing.put_text(image, name, vector.get_middle(), (255, 255, 255), 1)

        return image

    def get_rotated(self, rotations):
        rotated = self.deep_copy()
        rotated.rotate(rotations)
        return rotated


def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]



class PuzzleCollection:
    def __init__(self, pieces: [ExtractedPuzzle] = None):
        self.pieces = pieces

    def align_all(self):
        """finds the best angle for each puzzle, so that the edges are aligned to a grid"""
        progress_bar = ProgressBar(total=len(self.pieces), msg="aligning puzzles")
        for i, puzzle in enumerate(self.pieces):
            progress_bar.update()
            puzzle.align_to_grid()
        progress_bar.conclude()
        return


    def establish_notches(self):
        """finds the notches for each puzzle"""
        progress_bar = ProgressBar(total=len(self.pieces), msg="finding notches")
        for i, puzzle in enumerate(self.pieces):
            progress_bar.update()
            puzzle.find_notches()
        progress_bar.conclude()
        return

    def find_corners(self):
        """finds the corners of each puzzle"""
        progress_bar = ProgressBar(total=len(self.pieces), msg="finding corners")
        for i, puzzle in enumerate(self.pieces):
            try:
                progress_bar.update()
                puzzle.find_corners()
            except Exception as e:
                self.pieces.remove(puzzle)
                progress_bar.print_info(f"Removed puzzle #{i}, no corners found: {e}")
                image_processing.view_image(puzzle.image, title="puzzle")
        progress_bar.conclude()
        return

    def find_common_rotation(self):
        """checks if puzzles are rectangles, if so, aligns them to the same rotation"""
        #TODO
        #self.rectangular = True/False
        pass

    def __str__(self):
        str = ""
        for i, puzzle in enumerate(self.pieces):
            str += f"puzzle {i}\n"
            str += puzzle.__str__()
        return str

    def get_preview(self):
        return image_processing.images_to_image([puzzle.get_preview() for puzzle in self.pieces])
    def save_puzzles(self, path):
        self.for_each(lambda puzzle, i: image_processing.save_image(f"{path}_{i}.png", puzzle.image))

    def pickle(self, path=None):
        path = f"pickles/{date.today()}.pickle" if path is None else path
        with open(path, "wb") as file:
            pickle.dump(self, file)

    def for_each(self, func):
        if len(func.__code__.co_varnames) == 1:
            for puzzle in self.pieces:
                func(puzzle)
        else:
            for i, puzzle in enumerate(self.pieces):
                func(puzzle, i)

    @classmethod
    def unpickle(cls):
        latest = sorted(list_files("pickles"))[-1]
        path = f"pickles/{latest}"
        with open(path, "rb") as file:
            return pickle.load(file)

    def partition_by_notch_type(self, notch_type: NotchType):
        """Returns 2 Collections that contains copies of puzzles (First - desired notch type, Second - the rest)"""
        partition = reduce(
            lambda state, puzzle: state[notch_type in puzzle.notches.values()].append(puzzle.deep_copy())
                                  or state, self.pieces, ([], []))

        return PuzzleCollection(partition[1]), PuzzleCollection(partition[0])


def turn_into_binary(image, threshold=0.0):
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 255*threshold:
                result[i][j] = 255
    return result


def find_contours(mask):
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


def rotate(image, angle, enlarge=True):
    if enlarge:
        image = image_processing.enlarge_image(image, 1.5)
    height, width = image.shape[:2]
    center = (width//2,height//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image,rotation_matrix, (width,height))
    return rotated_image

def rotate_90(image, rotations):
    rotations = 4 - rotations #convert clockwise to counterclockwise
    return np.rot90(image, rotations)



def num_of_edges(mask):
    edges = cv2.Canny(mask, 10, 200)
    bounded, _ = image_processing.bound_image(edges)
    edges = bounded//255
    num = 0
    for line in edges:
        counts = np.bincount(line)

        if len(counts) == 2:
            if counts[1] > 0.3 * counts[0]:
                num += 1
    return num


def get_puzzles_from_masks(image, masks):
    puzzles = []
    for mask in masks:
        puzzle_image = image * mask[:, :, None]

        puzzle_image = bound_puzzle(puzzle_image, mask)
        mask = bound_puzzle(mask * 255, mask)
        extracted_puzzle = ExtractedPuzzle(image=puzzle_image, mask=mask)
        puzzles.append(extracted_puzzle)
    puzzle_collection = PuzzleCollection(puzzles)
    return puzzle_collection


def find_neighbours(values, distance=5):
    """
        groups values, which values are close to each other
        ex. [1,2,3,6,7,8,9] -> [[1,2,3],[6,7,8,9]]
    """
    result = []

    for v in values:
        subset = [r for r in values if abs(r - v) < distance]
        if subset not in result:
            result.append(subset)
    return result


def extract_puzzles(image, mask):
    print("extracting puzzles")
    masks = find_contours(mask)
    print(f"number of puzzles: {len(masks)}")
    puzzle_collection = get_puzzles_from_masks(image, masks)
    puzzle_collection.align_all()
    puzzle_collection.find_corners()
    puzzle_collection.establish_notches()

    return puzzle_collection




if __name__ == '__main__':
    name = "bliss"
    #name = "processed_photo"
    path = f"results/{name}.png"

    image = image_processing.load_image(path)
    mask = image_processing.load_image(path.replace(".", "_mask."))

    timer = timer.Timer()
    puzzle_collection = extract_puzzles(image, mask)
    timer.print("extracting puzzles")

    puzzle_collection.pickle()

    big_preview = puzzle_collection.get_preview()
    image_processing.save_image(f"extracted/{name}_log.png", big_preview)
    image_processing.view_image(big_preview, title="log")
    puzzle_collection.for_each(lambda puzzle: image_processing.view_image(puzzle.get_preview(), title="puzzle preview"))

