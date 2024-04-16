import random
from queue import PriorityQueue
import connecting
import image_processing
import puzzle_extracting
import puzzle_generator

#image = image_processing.load_image("input_photos/bliss.png")
#image_processing.view_image(image, "original image")
#
#puzzles, mask = puzzle_generator.image_to_puzzles(image=image, vertical_puzzle_size=3)
#image_processing.view_image(puzzles, "generated puzzle")

load_pickle = False


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

best = PriorityQueue()
i = 0


while True:
    edge1, edge2 = random.choice(edges), random.choice(edges)
    puzzle1_index, puzzle2_index = random.sample(range(30), 2)
    puzzle1, puzzle2 = puzzle_collection.pieces[puzzle1_index], puzzle_collection.pieces[puzzle2_index]
    if not connecting.is_connection_possible(puzzle1, edge1, puzzle2, edge2):
        continue
    result = connecting.connect_puzzles(puzzle1, edge1, puzzle2, edge2)
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
        result = connecting.connect_puzzles(puzzle1, edge1, puzzle2, edge2)
        edge_similarity, length_similarity, xor_img, and_img = result
        print(f"best similarity = {best_sim}, indexes = {index1, index2}, edges = {edge1, edge2}")
        preview = image_processing.images_to_image([xor_img, and_img])
        image_processing.view_image(preview, "best preview")




