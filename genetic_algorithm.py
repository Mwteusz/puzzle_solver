import math
import random
import numpy as np
from tqdm import tqdm

import image_processing
import puzzle_snake
from puzzle_extracting import ExtractedPuzzle, PuzzleCollection
from teeth_detection import NotchType
from teeth_detection import get_next_type
from teeth_detection import get_previous_type
from matching_puzzles import is_connection_possible, MatchException, number_of_rotations
from matching_puzzles import connect_puzzles

session_id = hash(random.random())

def edges_to_test(notches: dict):
    """get sides to compare the next piece with"""
    types = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    for type in types:
        notch_type = notches[type]
        if notch_type == NotchType.NONE:
            next_type = get_next_type(type)
            if notches[next_type] == NotchType.NONE:  # corner
                return get_next_type(next_type), get_previous_type(next_type)
            else:
                return get_next_type(type), get_previous_type(type)



def calculate_similarity(similarity, length_similarity, n=2):
    return (1 - (similarity + length_similarity) / 2) ** (1. / n)

#def calculate_similarity(similarity, length_similarity):
#    return 1 - (similarity + length_similarity) / 2

fit_cache = {}
def fitFun(puzzles, print_fits=False, get_fits=False):
    score = 0

    edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]

    fits = []
    for i, piece in enumerate(puzzles):
        next_piece= puzzles[(i+1)%len(puzzles)]

        edge1, edge2 = edges_to_test(piece.notches)

###########
        best_fit = 1
        best_edge = edges[0]

        for tested_edge in edges:

            try:
                if (piece.id, next_piece.id, edge1, tested_edge, next_piece.rotation, piece.rotation) in fit_cache:
                    add = fit_cache[(piece.id, next_piece.id, edge1, tested_edge, next_piece.rotation, piece.rotation)]
                    #print("cached:",len(fit_cache))
                else:
                    is_connection_possible(piece, edge1, next_piece, tested_edge)
                    similarity, length_similarity, img1, img2 = connect_puzzles(piece, edge1, next_piece, tested_edge)
                    add = calculate_similarity(similarity, length_similarity)
                    fit_cache[(piece.id, next_piece.id, edge1, tested_edge, next_piece.rotation, piece.rotation)] = add
            except MatchException:
                add = 1 # if the connection is not possible, the fit is the worst

            if add < best_fit:
                best_fit = add
                best_edge = tested_edge
        rotations = number_of_rotations(edge1, best_edge)
        next_piece.rotate(rotations)
###########
        score += best_fit
        if get_fits or print_fits:
            fit_string = f"id1:{piece.id}, rot1:{piece.rotation}, id2:{next_piece.id}, rot2:{next_piece.rotation}, fit:{add:.2f}"
            fits.append(fit_string)

    if print_fits:
        [print(fit) for fit in fits]
    if get_fits:
        return score, fits

    return score # 0 is the best fit (the least distance)


class Evolution:
    chromosomes = []
    num_of_chromosomes = 0
    num_of_genes = 0
    elitism_chance = 0.0
    mutation_rotate_chance = 0.0

    def __init__(self, chromosomes, genes, mutation_rotate, mutation_swap, elitism, do_rotate=True):
        self.num_of_genes = genes
        self.num_of_chromosomes = chromosomes
        self.mutation_rotate_chance = mutation_rotate
        self.mutation_swap_chance = mutation_swap
        self.elitism_chance = elitism
        self.rotate = do_rotate


        for i in range(num_of_chromosomes):
            if do_rotate:
                filtered_copy = [piece.get_rotated(random.randint(0, 3), False) for piece in edge_pieces]
            else:
                filtered_copy = [piece.deep_copy(False) for piece in edge_pieces]
            random.shuffle(filtered_copy)
            self.chromosomes.append(filtered_copy)

    def roulette(self, chromosomes):
        fitness = [1 / fitFun(x) for x in self.chromosomes]
        fitness = fitness / np.sum(fitness)

        new_chromosomes = []
        for chromosome in chromosomes:
            num = random.random()
            sum = 0
            index = 0
            for n in fitness:
                sum += n
                if num < sum:
                    new_chromosomes.append([piece.deep_copy(False) for piece in chromosome])
                    break
                index += 1

        return new_chromosomes
    def crossover(self, mother, father):

        mother_ids = [piece.id for piece in mother]
        father_ids = [piece.id for piece in father]

        a, b = random.sample(range(self.num_of_genes), 2)
        a, b = min(a, b), max(a, b) # get slice bounds

        mother_slice = mother_ids[a:b] # get ids of the slice
        son = [piece.deep_copy(False) for piece in father if (piece.id not in mother_slice)] # copy father's pieces that are not in the slice
        son[a:a] = [piece.deep_copy(False) for piece in mother if (piece.id in mother_slice)] # insert mother's slice into son

        father_slice = father_ids[a:b]
        daughter = [piece.deep_copy(False) for piece in mother if (piece.id not in father_slice)]
        daughter[a:a] = [piece.deep_copy(False) for piece in father if (piece.id in father_slice)]


        # rotate the slices
        #if self.rotate:
        #    if random.random() < self.mutation_rotate_chance:
        #        r = random.randint(1, 3)
        #        [piece.rotate(r) for piece in son if (piece.id in mother_slice)]
        #    if random.random() < self.mutation_rotate_chance:
        #        r = random.randint(1, 3)
        #        [piece.rotate(r) for piece in daughter if (piece.id in father_slice)]

        # print(f"---crossover---\n\t{a,b}\n\t{mother_ids}\n\t{father_ids}\n\t{son_ids}\n\t{daughter_ids}\n")

        return son, daughter

    def rotate_mutation(self, chromosomes, probability):
        if not self.rotate:
            return

        for chromosome in chromosomes:
            if random.random() < probability:
                random_piece = random.choice(chromosome)
                rotate_amount = random.randint(1, 3)
                random_piece.rotate(rotate_amount)

    def swap_mutation(self, chromosomes, probability):
        for chromosome in chromosomes:
            if random.random() < probability:
                index1, index2 = random.sample(range(self.num_of_genes), 2)
                chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]

    def insertion_mutation(self, chromosomes, probability):
        for chromosome in chromosomes:
            if random.random() < probability:
                index1, index2 = random.sample(range(self.num_of_genes), 2)
                chromosome.insert(index1, chromosome.pop(index2))

    def pivot_mutation(self, chromosomes, probability):
        for chromosome in chromosomes:
            if random.random() < probability:
                pivot = random.randint(0, self.num_of_genes - 1)
                chromosome[:] = chromosome[pivot:] + chromosome[:pivot]

    def elitism(self):
        """splits the population into elites and the rest of the population"""
        self.chromosomes.sort(key=fitFun, reverse=True)
        elites = []
        for i in range(int(self.elitism_chance * self.num_of_chromosomes)):
            elites.append(self.chromosomes[-i])

        rest = [chromosome for chromosome in self.chromosomes if (chromosome not in elites)]
        return elites, rest

    def iteration(self):
        self.chromosomes.sort(key=fitFun, reverse=True)
        best_chromosomes, temp_population = self.elitism()
        temp_population = self.roulette(self.chromosomes)[:len(temp_population)]
        children = []

        for i in range(0, len(temp_population), 2):
            parent1, parent2 = temp_population[i], temp_population[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            children += [child1, child2]

        self.rotate_mutation(children, self.mutation_rotate_chance)
        self.swap_mutation(children, self.mutation_swap_chance)
        self.chromosomes = best_chromosomes + children
        self.chromosomes.sort(key=fitFun, reverse=True)

    def get_best_chromosome(self):
        return self.chromosomes[-1]

    def __str__(self):
        result = ""
        for i, chromosome in enumerate(self.chromosomes):
            result += f"{i}: {fitFun(chromosome)}\n"
        return result

    def get_sum_of_fits(self):
        return sum(fitFun(chromosome) for chromosome in self.chromosomes)


def apply_images_to_puzzles(puzzles):
    for i, puzzle in enumerate(puzzles):
        puzzle.image = edge_pieces[puzzle.id].get_rotated(puzzle.rotation).image


edge_pieces = None


def save_snake(fitness_logs, snake_animation, iteration):
    max_width = max([image.shape[1] for image in snake_animation])
    max_height = max([image.shape[0] for image in snake_animation])

    snake_animation = [image_processing.expand_right_bottom(image, max_height, max_width) for image in snake_animation]
    path = f"snakes/session{session_id}/iteration{iteration}"

    for i, image in enumerate(snake_animation):
        image_processing.save_image(f"{path}/piece{i}.png", image)

    image_processing.save_gif(f"{path}/snake.gif", snake_animation)

    # save the fitness as txt
    with open(f"{path}/fitness.txt", "w") as file:
        file.write(f"sum of fits: {best_fit:.3f}\n")
        for log in fitness_logs:
            file.write(log + "\n")

    print(f"saved snake_it{iteration}")



if __name__ == '__main__':

    puzzle_collection = PuzzleCollection.unpickle()
    puzzle_collection, _ = puzzle_collection.partition_by_notch_type(NotchType.NONE)
    puzzle_collection.set_ids()
    #image_processing.view_image(puzzle_collection.get_preview(),"edge pieces")
    edge_pieces = puzzle_collection.pieces


    num_of_iterations = 10000000
    num_of_chromosomes = 100
    num_of_genes = len(edge_pieces)
    desired_fit = 0.5

    evolution = Evolution(num_of_chromosomes, num_of_genes, 0, 0.1, 0.2, do_rotate=True)

    record_fit = num_of_genes*2
    for it in tqdm(range(num_of_iterations)):
        evolution.iteration()

        best_chromosome = evolution.get_best_chromosome()
        best_fit, fitness_logs = fitFun(best_chromosome, get_fits=True)


        if it % 100 == 0:

            print(f" sum of fits: {evolution.get_sum_of_fits():.2f}", end=" ")
            print(f"best fit: {best_fit:.3f}", end=" ")
            print(f"piece ids: {[piece.id for piece in best_chromosome]}")


        if (best_fit < record_fit) or (it == num_of_iterations - 1) or (best_fit < desired_fit):
            record_fit = best_fit
            fitFun(best_chromosome, print_fits=False)
            print(f"best fit: {best_fit:.3f}")

            apply_images_to_puzzles(best_chromosome)
            snake_animation = puzzle_snake.get_snake_animation(best_chromosome, show_animation=False)
            #snake = puzzle_snake.get_snake_image(best_chromosome)
            #image_processing.view_image(image, f"fit={best_fit:.2f}, it={it}")

            save_snake(fitness_logs, snake_animation, it)
            puzzle_snake.snake_images = []

            if best_fit < desired_fit:
                answer = input("Do you want to continue? (y/n)")
                if answer.lower() == "n":
                    print("stopping...")
                    image_processing.view_image(snake_animation[-1], "final snake")
                    break
                else:
                    print("continuing...")
                    desired_fit -= 0.1
                    if desired_fit < 0:
                        desired_fit = 0
                    print(f"desired fit set to: {desired_fit:.3f}")



