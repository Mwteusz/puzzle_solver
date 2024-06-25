import random

import numpy as np

from matching_puzzles import connect_puzzles
from matching_puzzles import is_connection_possible, MatchException, number_of_rotations
from teeth_detection import get_next_type
from matching_puzzles import calculate_similarity

EDGES = ["TOP", "RIGHT", "BOTTOM", "LEFT"]
PREV_PIECE_EDGE = "RIGHT"
UPPER_PIECE_EDGE = "BOTTOM"

class InsideEvo:
    population = []

    def __init__(self, available_pieces, chromosomes, genes, pattern, mutation, elitism):
        self.available_pieces = available_pieces
        self.num_of_chromosomes = chromosomes
        self.num_of_genes = genes
        self.pattern = pattern
        self.mutation_rate = mutation
        self.elitism_chance = elitism
        self.do_clusters = False
        self.min_cluster_thresh = 0
        self.iteration_num = 0

        for i in range(self.num_of_chromosomes):
            pieces_copy = [piece.get_rotated(random.randint(0, 3), True) for piece in available_pieces]
            self.population.append(random.sample(pieces_copy, self.num_of_genes))

            for j in range(len(self.population[i])):
                self.population[i][j].cluster_id = j

    def fit_fun(self, chromosome, best_pair = False):
        score = 0
        best_pair_fitness = 1

        for i, piece in enumerate(chromosome):
            prev_piece = self.pattern[0] if i == 0 else chromosome[i - 1]
            upper_piece = self.pattern[i + 1]

            best_fit = 1
            best_edge = EDGES[0]

            for tested_edge in EDGES:
                try:
                    is_connection_possible(prev_piece, PREV_PIECE_EDGE, piece, tested_edge)
                    is_connection_possible(upper_piece, UPPER_PIECE_EDGE, piece, get_next_type(tested_edge))

                    similarity, length_similarity, image_similarity, _, _ = connect_puzzles(prev_piece, PREV_PIECE_EDGE, piece, tested_edge)
                    add = calculate_similarity(similarity, length_similarity, image_similarity) / 2

                    similarity, length_similarity, image_similarity, _, _ = connect_puzzles(prev_piece, PREV_PIECE_EDGE, piece, get_next_type(tested_edge))
                    add += calculate_similarity(similarity, length_similarity, image_similarity) / 2
                except MatchException:
                    add = 1

                if add < best_fit:
                    best_fit = add
                    best_edge = tested_edge

            rotations = number_of_rotations("LEFT", best_edge)
            piece.rotate(rotations)

            if self.do_clusters:
                if best_fit < self.min_cluster_thresh:
                    match_id = prev_piece.cluster_id
                    for p in filter(lambda _p: _p.cluster_id == match_id, chromosome):
                        p.cluster_id = piece.cluster_id

            score += best_fit

            if best_fit < best_pair_fitness:
                best_pair_fitness = best_fit

        if best_pair:
            return best_pair_fitness

        return score

    def roulette(self, chromosomes):
        fitness = [1 / self.fit_fun(x) for x in self.population]
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

        return son, daughter

    def swap_mutation(self):
        for chromosome in self.population:
            if random.random() < self.mutation_rate:
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


    def insert_mutation(self):
        for chromosome in self.population:
            if random.random() < self.mutation_rate:
                index = random.sample(range(len(self.population)))
                cluster_id = chromosome[index].cluster_id

                if chromosome[index-1].cluster_id == cluster_id:
                    continue

                if index != len(chromosome)-1 and chromosome[index+1].cluster_id == cluster_id:
                    continue

                existing_ids = list(map(lambda puzzle: puzzle.id, chromosome))
                available_puzzles = list(filter(lambda puzzle: puzzle.id not in existing_ids, self.available_pieces))
                piece = random.choice(available_puzzles).deep_copy()
                chromosome[index] = piece
                piece.cluster_id = cluster_id

    def elitism(self):
        """splits the population into elites and the rest of the population"""
        self.population.sort(key=self.fit_fun, reverse=True)
        elites = []
        for i in range(int(self.elitism_chance * self.num_of_chromosomes)):
            elites.append(self.population[-i])

        rest = [chromosome for chromosome in self.population if (chromosome not in elites)]
        return elites, rest


    def iteration(self):
        self.population.sort(key=self.fit_fun, reverse=True)
        if self.do_clusters or self.iteration_num > len(self.population[0]):
            current_best_fit = self.fit_fun(self.population[-1], best_pair=True) * 1.05
            if current_best_fit < self.min_cluster_thresh or self.min_cluster_thresh == 0:
                self.min_cluster_thresh = current_best_fit
                print(f"New threshold: {self.min_cluster_thresh}")
            self.do_clusters = True
        best_chromosomes, temp_population = self.elitism()
        temp_population = self.roulette(self.population)[:len(temp_population)]
        children = []

        for i in range(0, len(temp_population), 2):
            parent1, parent2 = temp_population[i], temp_population[i + 1]
            child1, child2 = self.crossover(parent1, parent2)
            children += [child1, child2]

        self.insert_mutation()
        self.swap_mutation()
        self.population = best_chromosomes + children
        self.population.sort(key=self.fit_fun, reverse=True)
        self.iteration_num += 1

    def get_best_chromosome(self):
        return self.population[-1]

    def __str__(self):
        result = ""
        for i, chromosome in enumerate(self.population):
            result += f"{i}: {self.fit_fun(chromosome)}\n"
        return result

    def get_sum_of_fits(self):
        return sum(self.fit_fun(chromosome) for chromosome in self.population)
