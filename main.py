
import image_processing
import puzzle_extracting
import scatter
import puzzle_generator

image = image_processing.load_image("input_photos/bliss.png")
puzzle_images, puzzle_masks = puzzle_generator.image_to_puzzles(image=image, vertical_puzzle_size=3)
preview = image_processing.images_to_image(puzzle_images)
image_processing.view_image(preview, "generated puzzle")

scattered_puzzle = scatter.scatter_pieces((image.shape[0]*2, image.shape[1]*2), pieces=puzzle_images, minimum_distance=10)
image_processing.view_image(scattered_puzzle, "scattered puzzle")

puzzle_collection = puzzle_extracting.extract_puzzles(scattered_puzzle)
image_processing.view_image(puzzle_collection.get_preview(), "extracted puzzles")


#puzzle_collection = PuzzleCollection.unpickle(name="2024-04-183x3.pickle")
#puzzle_collection.align_all() #rotates the pieces
#puzzle_collection.find_corners()
#puzzle_collection.establish_notches()




