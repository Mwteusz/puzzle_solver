import math
import random
import numpy as np
from tqdm import tqdm

import puzzle_snake
from puzzle_extracting import PuzzleCollection
from teeth_detection import NotchType
from teeth_detection import get_next_type
from teeth_detection import get_previous_type
from connecting import is_connection_possible
from connecting import connect_puzzles


def edges_to_test(notches):
    """get sides to compare the next piece with"""
    types = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
    for type in types:
        notch_type = notches[type]
        if notch_type == NotchType.NONE:
            next_type = get_next_type(type)
            if next_type == NotchType.NONE:  # corner
                return get_next_type(next_type), get_previous_type(next_type)
            else:
                return get_next_type(type), get_previous_type(type)


def fitFun(puzzles):
    score = 0

    for i, piece in enumerate(puzzles):
        if i == len(puzzles) - 1:
            break  # Why not compare last n first???
        next_piece = puzzles[i + 1]

        edge1, edge2 = edges_to_test(piece.notches)
        if is_connection_possible(piece, edge1, next_piece, edge2):
            similarity, length_similarity, img1, img2 = connect_puzzles(piece, edge1, next_piece, edge2)
            add = (similarity + length_similarity) / 2
            score += add

    max_potential_score = len(puzzles)
    return -score + max_potential_score


class Evolution:
    chromosomes = []
    num_of_chromosomes = 0
    num_of_genes = 0
    elitism_chance = 0.0
    mutation_rotate_chance = 0.0

    def __init__(self, chromosomes, genes, mutation_rotate, mutation_swap, elitism):
        self.num_of_genes = genes
        self.num_of_chromosomes = chromosomes
        self.mutation_rotate_chance = mutation_rotate
        self.mutation_swap_chance = mutation_swap
        self.elitism_chance = elitism

        for i in range(num_of_chromosomes):
            filtered_copy = [piece.deep_copy() for piece in filtered]
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
                    new_chromosomes.append([piece.deep_copy() for piece in chromosome])
                    break
                index += 1

        return new_chromosomes

    #def roulette(self, chromosomes):
    #    fitnesses = [fitFun(chromosome) for chromosome in chromosomes]
    #    sum = np.sum(fitnesses)
    #    if sum == 0:
    #        return
    #    fitnesses = np.array(fitnesses) / sum
#
    #    indexes = range(len(chromosomes))
    #    new_chromosomes = []
    #    for i in range(len(chromosomes)):
    #        index = np.random.choice(indexes, p=fitnesses)
    #        new_chromosomes.append([piece.deep_copy() for piece in chromosomes[index]])
#
    #    return new_chromosomes

    def crossover(self, mother, father):

        mother_ids = [piece.id for piece in mother]
        father_ids = [piece.id for piece in father]

        a, b = random.sample(range(self.num_of_genes), 2)
        a, b = min(a, b), max(a, b)

        slice1 = mother_ids[a:b]
        son_ids = [id for id in father_ids if (id not in slice1)]
        son_ids[a:a] = slice1

        slice2 = father_ids[a:b]
        daughter_ids = [id for id in mother_ids if (id not in slice2)]
        daughter_ids[a:a] = slice2

        son = [filtered[id].deep_copy() for id in son_ids]
        daughter = [filtered[id].deep_copy() for id in daughter_ids]

        return son, daughter

    def rotate_mutation(self, chromosomes, probability):
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

        # for chromosome in chromosomes:
        #    if random.random() < probability:
        #        index = random.randint(0, self.num_of_genes - 2)
        #        chromosome[index], chromosome[index+1] = chromosome[index+1], chromosome[index]

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
        temp_population = self.roulette(temp_population)
        children = []

        for i in range(0, len(temp_population), 2):
            parent1, parent2 = temp_population[i], temp_population[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            children += [child1, child2]

        self.rotate_mutation(children, self.mutation_rotate_chance)
        self.swap_mutation(children, self.mutation_swap_chance)
        self.chromosomes = best_chromosomes + children
        self.chromosomes.sort(key=fitFun, reverse=True)

    def __str__(self):
        result = ""
        for i, chromosome in enumerate(self.chromosomes):
            result += f"{i}: {fitFun(chromosome)}\n"
        return result


filtered = None
if __name__ == '__main__':

    puzzle_collection = PuzzleCollection.unpickle()
    filtered, _ = puzzle_collection.partition_by_notch_type(NotchType.NONE)
    filtered = filtered.pieces

    for i, piece in enumerate(filtered):
        piece.id = i

    num_of_chromosomes = 100
    num_of_genes = len(filtered)

    # cities = [i for i in range(25)]

    # points = [(119, 38), (37, 38), (197, 55), (85, 165), (12, 50), (100, 53), (81, 142), (121, 137), (85, 145),
    #          (80, 197), (91, 176), (106, 55), (123, 57), (40, 81), (78, 125), (190, 46), (187, 40), (37, 107),
    #          (17, 11), (67, 56), (78, 133), (87, 23), (184, 197), (111, 12), (66, 178)]

    evolution = Evolution(num_of_chromosomes, num_of_genes, 0.02, 0.01,0.2)
    for it in tqdm(range(1000)):
        evolution.iteration()
        if it % 100 == 0:
            best_chromosome = evolution.chromosomes[-1]
            print(f" iteration {it}, best: {fitFun(best_chromosome)}")

        #print(evolution)

    best_chromosome = evolution.chromosomes[-1]
    #worst_chromosome = evolution.chromosomes[0]
    # print(fitFun(best_chromosome))


    for i, chromosome in enumerate(evolution.chromosomes):
        print(i, fitFun(chromosome))
        #image = puzzle_snake.get_snake_image(chromosome)
        #puzzle_snake.image_processing.view_image(image,fitFun(chromosome))
    image = puzzle_snake.get_snake_image(best_chromosome)
    puzzle_snake.image_processing.view_image(image, fitFun(best_chromosome))
