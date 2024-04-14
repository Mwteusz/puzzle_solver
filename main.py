import image_processing
import puzzle_extracting
import puzzle_generator

image = image_processing.load_image("input_photos/bliss.png")
image_processing.view_image(image, "original image")

puzzles, mask = puzzle_generator.image_to_puzzles(image=image, vertical_puzzle_size=5)
image_processing.view_image(puzzles, "generated puzzle")

puzzle_collection = puzzle_extracting.extract_puzzles(puzzles, mask)
output = puzzle_collection.get_preview()
image_processing.view_image(output, "puzzle preview")

image_processing.save_image("results/the_grand_output.png", output)


