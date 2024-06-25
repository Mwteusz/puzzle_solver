import math
import random
import numpy as np
from tqdm import tqdm

import image_processing
import puzzle_snake
from genetic_algorithm_pt2 import InsideEvo
from puzzle_extracting import ExtractedPuzzle, PuzzleCollection
from teeth_detection import NotchType
from teeth_detection import get_next_type
from teeth_detection import get_previous_type
from matching_puzzles import is_connection_possible, MatchException, number_of_rotations
from matching_puzzles import connect_puzzles
from matching_puzzles import calculate_similarity

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







#def calculate_similarity(similarity, length_similarity):
#    return 1 - (similarity + length_similarity) / 2

#### global

#### ####



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
        self.iteration_num = 0
        self.fit_cache = {}
        self.do_clusters = False
        self.min_cluster_thresh = 0
        self.default_mutation_chance = mutation_swap

        for i in range(num_of_chromosomes):
            if do_rotate:
                filtered_copy = [piece.get_rotated(random.randint(0, 3), True) for piece in edge_pieces]
            else:
                filtered_copy = [piece.deep_copy(True) for piece in edge_pieces]
            for j in range(len(filtered_copy)):
                filtered_copy[j].cluster_id = j
            random.shuffle(filtered_copy)
            self.chromosomes.append(filtered_copy)

        self.last_best_fit = self.fitFun(self.chromosomes[0], return_best_fit=True)

    def fitFun(self, puzzles, print_fits=False, get_fits=False, return_best_fit=False, return_best_fits=False):
        score = 0

        edges = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
        bestbest_fit = 1
        fits = []
        best_fits = []
        for i, piece in enumerate(puzzles):
            next_piece = puzzles[(i + 1) % len(puzzles)]

            edge1, edge2 = edges_to_test(piece.notches)

            if piece.cluster_id == next_piece.cluster_id:
                try:
                    if (piece.id, next_piece.id, edge1, edge2, next_piece.rotation, piece.rotation) in self.fit_cache:
                        add = self.fit_cache[(piece.id, next_piece.id, edge1, edge2, next_piece.rotation, piece.rotation)]
                        # print("cached:",len(fit_cache))
                    else:
                        is_connection_possible(piece, edge1, next_piece, edge2)
                        similarity, length_similarity, image_similarity, img1, img2 = connect_puzzles(piece, edge1,
                                                                                                      next_piece, edge2)
                        add = calculate_similarity(similarity, length_similarity, image_similarity)
                        # add = image_similarity
                        self.fit_cache[(piece.id, next_piece.id, edge1, edge2, next_piece.rotation, piece.rotation)] = add
                except MatchException:
                    add = 1
                score += add
                continue

            ###########
            best_fit = 1
            best_edge = edges[0]

            for tested_edge in edges:

                try:
                    if (piece.id, next_piece.id, edge1, tested_edge, next_piece.rotation, piece.rotation) in self.fit_cache:
                        add = self.fit_cache[
                            (piece.id, next_piece.id, edge1, tested_edge, next_piece.rotation, piece.rotation)]
                        # print("cached:",len(fit_cache))
                    else:
                        is_connection_possible(piece, edge1, next_piece, tested_edge)
                        similarity, length_similarity, image_similarity, img1, img2 = connect_puzzles(piece, edge1,
                                                                                                      next_piece,
                                                                                                      tested_edge)
                        add = calculate_similarity(similarity, length_similarity, image_similarity)
                        # add = image_similarity
                        self.fit_cache[
                            (piece.id, next_piece.id, edge1, tested_edge, next_piece.rotation, piece.rotation)] = add
                except MatchException:
                    add = 1  # if the connection is not possible, the fit is the worst

                if add < best_fit:
                    best_fit = add
                    best_edge = tested_edge
            rotations = number_of_rotations(edge1, best_edge)

            # check for clusters for next piece
            next_piece_cluster_id = next_piece.cluster_id
            for p in filter(lambda _p: _p.cluster_id == next_piece_cluster_id, puzzles):
                p.rotate(rotations)

            # for pieceMini in puzzles:
            #     if pieceMini.cluster_id == next_piece_cluster_id:
            #         pieceMini.rotate(rotations)

            ## next_piece.rotate(rotations)
            ###########
            if self.do_clusters:
                if best_fit < self.min_cluster_thresh:
                    match_id = next_piece.cluster_id
                    for p in filter(lambda _p: _p.cluster_id == match_id, puzzles):
                        p.cluster_id = piece.cluster_id

            score += best_fit
            if best_fit < bestbest_fit:
                bestbest_fit = best_fit
            if get_fits or print_fits or return_best_fits:
                best_fits.append(best_fit)
                fit_string = f"id1:{piece.id}, rot1:{piece.rotation}, id2:{next_piece.id}, rot2:{next_piece.rotation}, best_fit:{best_fit:.2f}"
                fits.append(fit_string)

        if print_fits:
            [print(fit) for fit in fits]
        if return_best_fit:
            return bestbest_fit
        if get_fits:
            return score, fits
        if return_best_fits:
            return best_fits

        return score  # 0 is the best fit (the least distance)

    def roulette(self, chromosomes):
        fitness = [1 / self.fitFun(x) for x in self.chromosomes]
        fitness = fitness / np.sum(fitness)

        new_chromosomes = []
        for chromosome in chromosomes:
            num = random.random()
            sum = 0
            index = 0
            for n in fitness:
                sum += n
                if num < sum:
                    new_chromosomes.append([piece.deep_copy(True) for piece in chromosome])
                    break
                index += 1

        return new_chromosomes
    def crossover(self, mother, father):

        possible_pivots = [idx for idx in range(len(mother) - 1) if mother[idx].cluster_id != mother[idx + 1].cluster_id]
        if len(possible_pivots) <= 1:
            return mother, father

        a, b = random.sample(possible_pivots, 2)
        a, b = min(a, b), max(a, b)

        mother_ids = [piece.id for piece in mother]
        father_ids = [piece.id for piece in father]

        mother_ids_slice = mother_ids[a:b]
        mother_slice = [piece.deep_copy(True) for piece in mother if (piece.id in mother_ids_slice)]
        mother_cluster_ids_slice = list(map(lambda piece: piece.cluster_id, mother_slice))
        son = [piece.deep_copy(True) for piece in father if (piece.id not in mother_ids_slice)]
        possible_son_ids = list(filter(lambda cluster_id: cluster_id not in mother_cluster_ids_slice, range(len(mother))))
        for son_pair in zip(son, possible_son_ids):
            son_pair[0].cluster_id = son_pair[1]
        son[a:a] = mother_slice



        possible_pivots = [idx for idx in range(len(father) - 1) if
                           father[idx].cluster_id != father[idx + 1].cluster_id]
        a, b = random.sample(possible_pivots, 2)
        a, b = min(a, b), max(a, b)

        father_ids_slice = father_ids[a:b]
        father_slice = [piece.deep_copy(True) for piece in father if (piece.id in father_ids_slice)]
        father_cluster_ids_slice = list(map(lambda piece: piece.cluster_id, father_slice))
        daughter = [piece.deep_copy(True) for piece in father if (piece.id not in father_ids_slice)]
        possible_daughter_ids = list(filter(lambda cluster_id: cluster_id not in father_cluster_ids_slice, range(len(father))))
        for daughter_pair in zip(daughter, possible_daughter_ids):
            daughter_pair[0].cluster_id = daughter_pair[1]
        daughter[a:a] = father_slice

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
                index1, index2 = random.sample(range(len(chromosome)), 2)
                id1 = chromosome[index1].cluster_id
                id2 = chromosome[index2].cluster_id

                if id1 == id2:
                    continue

                ids1 = list(filter(lambda puzzle: puzzle.cluster_id == id1, chromosome))
                ids2 = list(filter(lambda puzzle: puzzle.cluster_id == id2, chromosome))
                rest = list(filter(lambda puzzle: puzzle.cluster_id != id1 and puzzle.cluster_id != id2, chromosome))

                idx1 = next(idx for idx, puzzle in enumerate(chromosome) if puzzle.cluster_id == id1)
                idx2 = next(idx for idx, puzzle in enumerate(chromosome) if puzzle.cluster_id == id2)

                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                    ids1, ids2 = ids2, ids1

                idx2 -= len(ids1) - len(ids2)

                swapped = [None] * len(chromosome)
                swapped[:0] = rest
                swapped[idx1:idx1] = ids2
                swapped[idx2:idx2] = ids1
                swapped = swapped[:len(chromosome)]

                for i in range(len(chromosome)):
                    chromosome[i] = swapped[i]

                # chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]

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
        self.chromosomes.sort(key=self.fitFun, reverse=True)
        elites = []
        for i in range(int(self.elitism_chance * self.num_of_chromosomes)):
            elites.append(self.chromosomes[-i])

        rest = [chromosome for chromosome in self.chromosomes if (chromosome not in elites)]
        return elites, rest

    def iteration(self):

        self.chromosomes.sort(key=self.fitFun, reverse=True)

        if self.fitFun(self.chromosomes[-1], return_best_fit=True) < self.last_best_fit:
            self.mutation_swap_chance = self.default_mutation_chance
        elif self.mutation_swap_chance < 0.4:
            self.mutation_swap_chance += 0.001

        if not self.do_clusters and self.iteration_num > len(self.chromosomes[0]):
            current_best_fit = self.fitFun(self.chromosomes[-1], return_best_fit=True) * 1.05
            if current_best_fit < self.min_cluster_thresh or self.min_cluster_thresh == 0:
                self.min_cluster_thresh = current_best_fit
            # self.min_cluster_thresh = self.fitFun(self.chromosomes[-1], return_best_fit=True) * 1.05
                print(f"New threshold: {self.min_cluster_thresh}")
            self.do_clusters = True
            best_fits = self.fitFun(self.chromosomes[-1], return_best_fits=True)
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
        self.chromosomes.sort(key=self.fitFun, reverse=True)
        self.iteration_num += 1

    def get_best_chromosome(self):
        return self.chromosomes[-1]

    def __str__(self):
        result = ""
        for i, chromosome in enumerate(self.chromosomes):
            result += f"{i}: {self.fitFun(chromosome)}\n"
        return result

    def get_sum_of_fits(self):
        return sum(self.fitFun(chromosome) for chromosome in self.chromosomes)


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


def is_completed(chromosome):
    main_cluster = chromosome[0].cluster_id
    return all(puzzle.cluster_id == main_cluster for puzzle in chromosome)


if __name__ == '__main__':

    #puzzle_collection = PuzzleCollection.unpickle("2024-04-28_scattered_bliss_v=3_r=False.pickle")
    puzzle_collection = PuzzleCollection.unpickle()
    puzzle_collection, inside_puzzle_collection = puzzle_collection.partition_by_notch_type(NotchType.NONE)
    puzzle_collection.set_ids()
    inside_puzzle_collection.set_ids()
    #image_processing.view_image(puzzle_collection.get_preview(),"edge pieces")
    edge_pieces = puzzle_collection.pieces
    inside_pieces = inside_puzzle_collection.pieces

    num_of_iterations = 1000000
    num_of_chromosomes = 100
    num_of_genes = len(edge_pieces)
    desired_fit = 0.5

    evolution = Evolution(num_of_chromosomes, num_of_genes, 0, 0.1, 0.2, do_rotate=True)

    record_fit = num_of_genes*2
    for it in tqdm(range(num_of_iterations)):
        evolution.iteration()

        best_chromosome = evolution.get_best_chromosome()
        best_fit, fitness_logs = evolution.fitFun(best_chromosome, get_fits=True)


        if it % 100 == 0:

            print(f" sum of fits: {evolution.get_sum_of_fits():.2f}", end=" ")
            print(f"best fit: {best_fit:.3f}", end=" ")
            print(f"piece ids: {[(piece.id, piece.cluster_id) for piece in best_chromosome]}")


        if (best_fit < record_fit) or (it == num_of_iterations - 1) or (best_fit < desired_fit):
            record_fit = best_fit
            evolution.fitFun(best_chromosome, print_fits=False)
            print(f"best fit: {best_fit:.3f}")

            apply_images_to_puzzles(best_chromosome)
            snake_animation = puzzle_snake.get_snake_animation(best_chromosome, show_animation=False)
            #snake = puzzle_snake.get_snake_image(best_chromosome)
            #image_processing.view_image(image, f"fit={best_fit:.2f}, it={it}")

            save_snake(fitness_logs, snake_animation, it)
            puzzle_snake.snake_images = []

        if is_completed(best_chromosome):
            break

    best_chromosome = evolution.get_best_chromosome()

    shift_value = list(puzzle.notches["TOP"] == NotchType.NONE and puzzle.notches["LEFT"] == NotchType.NONE for puzzle in best_chromosome).index(True)
    shifted_chromosome = best_chromosome[shift_value:] + best_chromosome[:shift_value]

    width = len(list(map(lambda puzzle: puzzle.notches["TOP"] == NotchType.NONE, best_chromosome)))
    height = len(list(map(lambda puzzle: puzzle.notches["LEFT"] == NotchType.NONE, best_chromosome)))

    pattern_matrix = np.empty((height, width), dtype=ExtractedPuzzle) * None

    tops = list(filter(lambda puzzle: puzzle.notches["TOP"] == NotchType.NONE, shifted_chromosome))
    rights = list(filter(lambda puzzle: puzzle.notches["RIGHT"] == NotchType.NONE, shifted_chromosome))
    bottoms = list(filter(lambda puzzle: puzzle.notches["BOTTOM"] == NotchType.NONE, shifted_chromosome))
    lefts = list(filter(lambda puzzle: puzzle.notches["LEFT"] == NotchType.NONE, shifted_chromosome))

    bottoms.reverse()
    lefts.reverse()

    for i in range(width):
        pattern_matrix[0][i] = tops[i]
        pattern_matrix[-1][i] = bottoms[i]

    for i in range(height):
        pattern_matrix[i][0] = lefts[i]
        pattern_matrix[i][-1] = rights[i]

    image_processing.show_image_matrix(pattern_matrix)

    number_of_genes = width - 2

    inside_evo = InsideEvo(inside_pieces, num_of_chromosomes, number_of_genes, [], 0.1, 0.2)

    for it in tqdm(range(num_of_iterations)):
        inside_evo.iteration()
