import random

import connecting
import image_processing
import photo_processing
import puzzle_extracting
import puzzle_generator

#image = image_processing.load_image("input_photos/bliss.png")
#image_processing.view_image(image, "original image")
#
#puzzles, mask = puzzle_generator.image_to_puzzles(image=image, vertical_puzzle_size=3)
#image_processing.view_image(puzzles, "generated puzzle")

load_pickle = True


if load_pickle is True:
    puzzle_collection = puzzle_extracting.PuzzleCollection.unpickle()
else:
    name = "processed_photo"
    path = f"results/{name}.png"
    image = image_processing.load_image(path)
    mask = image_processing.load_image(path.replace(".", "_mask."))

    puzzle_collection = puzzle_extracting.extract_puzzles(image, mask)
    puzzle_collection.pickle()

output = puzzle_collection.get_preview()
image_processing.view_image(output, "puzzle preview")
image_processing.save_image("results/the_grand_output.png", output)


edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
while True:
    edge1, edge2 = random.choice(edges), random.choice(edges)
    puzzle1_index, puzzle2_index = random.sample(range(30), 2)
    puzzle1, puzzle2 = puzzle_collection.pieces[puzzle1_index], puzzle_collection.pieces[puzzle2_index]
    if not connecting.is_connection_possible(puzzle1, edge1, puzzle2, edge2):
        continue
    result = connecting.connect_puzzles(puzzle1, edge1, puzzle2, edge2)
    similarity, length_similarity, output_img = result
    print(f"puzzle1_index = {puzzle1_index}, puzzle2_index = {puzzle2_index}, edge1 = {edge1}, edge2 = {edge2}")
    print(f"similarity = {similarity}, length_similarity = {length_similarity}")
    image_processing.view_image(output_img, "output")



