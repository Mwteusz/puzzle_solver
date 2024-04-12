import cv2
import numpy as np
import image_processing
import teeth_detection
from teeth_detection import NotchType


class ExtractedPuzzle:
    def __init__(self, notches: [NotchType] = None, image = None, mask = None ,corners = None):
        self.notches = notches
        self.image = image
        self.mask = mask
        self.corners = corners

    def __str__(self):
        str = ""
        for type, result in self.notches.items():
            str += f"{type}\t {result}\n"
        return str

    def rotate(self, direction):
        """ direction means 'left' or 'right' """
        #TODO
        #rotate image,
        #move the notches over,
        #done, return None
        pass

    def find_corners(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        edges = cv2.Canny(self.mask * 255, 10, 200)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
        edges, point = bound_image(edges)
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

        for a1, a2 in corners:
            for b1, b2 in corners:
                field = min(abs(a1[0] - a2[0]), abs(b1[0] - b2[0])) * abs(a1[1] - b2[1])
                if field > max_field:
                    max_field = field
                    best_corners = [a1, a2, b1, b2]
        self.corners = [(corner[0] + point[0], corner[1] + point[1]) for corner in best_corners]

    def get_preview(self):
        #display edges
        preview = self.image.copy()
        for type, vector in teeth_detection.get_vectors_from_corners(self.corners).items():
            cv2.line(preview, vector.point1, vector.point2, (255,128,64), 1)
            notch_type = self.notches[type]
            name = notch_type.name.removeprefix("NotchType.")

            cv2.putText(preview, f"{name}", vector.get_point_between(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        for corner in self.corners:
            cv2.circle(preview, corner, 5, (255,0,0), -1)

        return preview


class PuzzleCollection:
    def __init__(self, pieces: [ExtractedPuzzle] = None):
        self.pieces = pieces

    def align_all(self):
        """finds the best angle for each puzzle, so that the edges are aligned to a grid"""
        for i, puzzle in enumerate(self.pieces):
            puzzle_image, mask = puzzle.image, puzzle.mask
            angles = []
            for angle in range(0, 360):
                rotated_mask = rotate(mask * 255, angle)
                num = num_of_edges(rotated_mask * 255)
                if num > 1:
                    angles.append(angle)
            if len(angles) == 0:
                raise Exception("No edges found!!!!!!!")

            groups = find_neighbours(angles)
            largest_group = max(groups, key=len)
            median_element = np.median(largest_group)

            puzzle.image = rotate(puzzle_image, median_element)
            puzzle.mask = rotate(mask, median_element) * 255
        return


    def establish_notches(self):
        """finds the notches for each puzzle"""
        for puzzle in self.pieces:
            puzzle.notches = teeth_detection.get_teeth(puzzle.image, puzzle.corners)

    def find_corners(self):
        """finds the corners of each puzzle"""
        for puzzle in self.pieces:
            puzzle.find_corners()
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


def turn_into_binary(image):
    if len(image.shape) == 2:
        return image

    result = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] > 0:
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


def rotate(image, angle):
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
    bounded, _ = bound_image(edges)
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


def find_neighbours(angles, distance=5):
    """
        groups values, which values are close to each other
        ex. [1,2,3,6,7,8,9] -> [[1,2,3],[6,7,8,9]]
    """
    result = []

    for angle in angles:
        subset = [r for r in angles if abs(r - angle) < distance]
        if subset not in result:
            result.append(subset)
    return result


def extract_puzzles(image, mask):
    masks = find_contours(mask)
    print(f"number of puzzles: {len(masks)}")
    puzzle_collection = get_puzzles_from_masks(image, masks)
    puzzle_collection.align_all()
    puzzle_collection.find_corners()
    puzzle_collection.establish_notches()

    return puzzle_collection



if __name__ == '__main__':
    path = "results/processed_photo.png"
    #path = "results/generated.png"

    image = image_processing.load_image(path)
    mask = image_processing.load_image(path.replace(".", "_mask."))

    puzzle_collection = extract_puzzles(image, mask)

    for i, puzzle in enumerate(puzzle_collection.pieces):
        print(puzzle)
        puzzle_name = f"puzzle_{i}"
        image_processing.view_image(puzzle.get_preview(), title=puzzle_name)
        image_processing.save_image(f"extracted/{puzzle_name}.png", puzzle.image)


