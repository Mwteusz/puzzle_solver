import random
from queue import PriorityQueue
import matching_puzzles
import image_processing
import puzzle_extracting
import puzzle_generator


image = image_processing.load_image("input_photos/widzew.png")
image_processing.view_image(image, "original image")

#puzzles, mask = puzzle_generator.image_to_puzzles(image=image, vertical_puzzle_size=3)
#image_processing.view_image(puzzles, "generated puzzle")
pieces, _ = puzzle_generator.create_puzzles(image, image.shape[0] // 3)
pieces = [piece.puzzle_image for piece in pieces]
masks = [image_processing.threshold(piece, 0) for piece in pieces]


load_pickle = False


if load_pickle is True:
    puzzle_collection = puzzle_extracting.PuzzleCollection.unpickle()
else:
    name = "3x3"
    path = f"results/{name}.png"
    #image = puzzles

    #puzzle_collection = puzzle_extracting.extract_puzzles(image, mask)
    puzzles = []
    for i, (image, mask) in enumerate(zip(pieces, masks)):
        puzzle = puzzle_extracting.ExtractedPuzzle(image=image, mask=mask, id=i)
        puzzles.append(puzzle)
    puzzle_collection = puzzle_extracting.PuzzleCollection(puzzles)

    #puzzle_collection.align_all()
    puzzle_collection.find_corners()
    puzzle_collection.establish_notches()
    image_processing.view_image(puzzle_collection.get_preview())

    puzzle_collection.pickle(suffix=name)

output = puzzle_collection.get_preview()
image_processing.view_image(output, "puzzle preview")
image_processing.save_image("results/the_grand_output.png", output)


edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]

best = PriorityQueue()
i = 0


puzzle_amount = len(puzzle_collection.pieces)
while True:
    edge1, edge2 = random.choice(edges), random.choice(edges)
    puzzle1_index, puzzle2_index = random.sample(range(puzzle_amount), 2)
    puzzle1, puzzle2 = puzzle_collection.pieces[puzzle1_index], puzzle_collection.pieces[puzzle2_index]
    if not matching_puzzles.is_connection_possible(puzzle1, edge1, puzzle2, edge2):
        continue
    result = matching_puzzles.connect_puzzles(puzzle1, edge1, puzzle2, edge2)
    edge_similarity, length_similarity, xor_img, and_img = result
    similarity = (edge_similarity + length_similarity)/2
    print(f"similarity = {similarity}, indexes = {puzzle1_index, puzzle2_index}, edges = {edge1, edge2}")

    queue_element = (puzzle1_index, puzzle2_index, edge1, edge2)
    best.put((similarity, queue_element))
    if len(best.queue) > 4:
        best.get()

    i+=1
    if i % 1000 == 0:
        print(f"iteration {i}")
        for i, (similarity, queue_element) in enumerate(best.queue):
            print(f"\t{i+1}. similarity = {similarity}, indexes = {queue_element[:2]}, edges = {queue_element[2:]}")
        #view best
        best_sim, best_element = best.queue[0]
        index1, index2, edge1, edge2 = best_element
        puzzle1, puzzle2 = puzzle_collection.pieces[index1], puzzle_collection.pieces[index2]
        result = matching_puzzles.connect_puzzles(puzzle1, edge1, puzzle2, edge2)
        edge_similarity, length_similarity, xor_img, and_img = result
        print(f"best similarity = {best_sim}, indexes = {index1, index2}, edges = {edge1, edge2}")
        preview = image_processing.images_to_image([xor_img, and_img])
        image_processing.view_image(preview, "best preview")




