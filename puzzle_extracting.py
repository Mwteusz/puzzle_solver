import timer
import cv2
import numpy as np
import image_processing
import teeth_detection
from progress_bar import ProgressBar
from teeth_detection import NotchType


def view_corners(image, corners):
    preview = image.copy()
    for corner in corners:
        cv2.circle(preview, corner, 5, (255, 0, 0), -1)
    image_processing.view_image(preview, title="corners")


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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.Canny(self.mask, 10, 200)
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


        #test_corners = []
        for a1, a2 in corners:
            #test_corners.append((a1[0] + point[0], a1[1] + point[1]))
            #test_corners.append((a2[0] + point[0], a2[1] + point[1]))
            for b1, b2 in corners:
                field = min(abs(a1[0] - a2[0]), abs(b1[0] - b2[0])) * abs(a1[1] - b2[1])
                if field > max_field:
                    max_field = field
                    best_corners = [a1, a2, b1, b2]

        self.corners = [(corner[0] + point[0], corner[1] + point[1]) for corner in best_corners]

        #view_corners(self.image, test_corners)
        #image_processing.view_image(edges, title="edges")
        if len(self.corners) != 4:
            raise Exception(f"4 corners should be found, but found: {len(self.corners)}")

    def align_to_grid(self):
        puzzle_image, mask = self.image, self.mask
        #optimization, may not be needed
        w, h = mask.shape
        if w > 100 or h > 100:
            mask = cv2.resize(mask, (100, 100))
        #
        angles = []

        enlarged = image_processing.enlarge_image(mask, 1.5) #cache enlarged image
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
        self.image = rotate(puzzle_image, median_element)
        self.corners = None  # corners need to be recalculated
        return
    def find_notches(self):
        self.notches = teeth_detection.get_teeth(self.image, self.corners)

    def get_preview(self):
        if self.corners is None or self.notches is None:
            raise Exception("corners or notches not found")

        preview = self.image.copy()
        for corner in self.corners:
            cv2.circle(preview, corner, 5, (255,0,0), -1)

        for type, vector in teeth_detection.get_vectors_from_corners(self.corners).items():
            cv2.line(preview, vector.point1, vector.point2, (255,128,64), 1)
            notch_type = self.notches[type]
            if notch_type is not NotchType.NONE:
                name = notch_type.name.removeprefix("NotchType.").capitalize()
                image_processing.put_text(preview, name, vector.get_point_between(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                image_processing.put_text(preview, name, vector.get_point_between(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return preview


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
                print(f"Could not find corners in puzzle #{i}: {e}")
                self.pieces.remove(puzzle)
                print(f"removed puzzle {i}!!!!!")
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
        return images_to_image([puzzle.get_preview() for puzzle in self.pieces])

    def for_each(self, func):
        for puzzle in self.pieces:
            func(puzzle)


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



def images_to_image(images):
    images = [cv2.resize(image_processing.square(image), (200, 200)) for image in images]
    size = int(np.ceil(np.sqrt(len(images))))
    image_size = images[0].shape[0]
    image_array = np.zeros((image_size*size, image_size*size,3), dtype=np.uint8)
    for i, image in enumerate(images):
        x = i % size
        y = i // size
        start_x = x * image_size
        start_y = y * image_size
        image_array[start_y:start_y+image_size, start_x:start_x+image_size] = image
    return image_array





if __name__ == '__main__':
    #name = "bliss"
    name = "processed_photo"
    path = f"results/{name}.png"

    image = image_processing.load_image(path)
    mask = image_processing.load_image(path.replace(".", "_mask."))

    timer = timer.Timer()
    puzzle_collection = extract_puzzles(image, mask)
    timer.print("extracting puzzles")

    puzzle_collection.for_each(lambda puzzle: image_processing.view_image(puzzle.get_preview(), title="puzzle preview"))

    big_preview = puzzle_collection.get_preview()
    image_processing.save_image(f"extracted/{name}_log.png", big_preview)
    image_processing.view_image(big_preview, title="log")

