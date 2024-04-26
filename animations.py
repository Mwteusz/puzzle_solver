import random

import image_processing
from puzzle_extracting import PuzzleCollection, ExtractedPuzzle

if __name__ == '__main__':
    puzzle_collection = PuzzleCollection.unpickle("2024-04-24_scattered_widzew_3x3_no_rotate.pickle")
    puzzle_collection.establish_notches()

    #random_piece = puzzle_collection.pieces[random.randint(0, len(puzzle_collection.pieces)-1)]
    #image_processing.view_image(random_piece.image)
