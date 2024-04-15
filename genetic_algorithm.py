import random
import numpy as np
from tqdm import tqdm

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
            if next_type == NotchType.NONE: # corner
                return get_next_type(next_type), get_previous_type(next_type)
            else:
                return get_next_type(type), get_previous_type(type)


def fitFun(puzzles):
    score = 0

    for i, piece in enumerate(puzzles):
        print(i)
        if i == len(puzzles)-1:
            break # Why not compare last n first???
        next_piece = puzzles[i+1]

        edge1, edge2 = edges_to_test(piece.notches)
        if is_connection_possible(piece,edge1, next_piece,edge2):
            similarity, length_similarity, output_img = connect_puzzles(piece, edge1, next_piece, edge2)
            score -= (similarity + length_similarity)
        else:
            score += 2

    return score

    # distance = 0

    # for i in range(len(x) - 1):
    #     distance += np.sqrt((points[x[i]][0] - points[x[i+1]][0])**2 + (points[x[i]][1] - points[x[i+1]][1])**2)
    # distance += np.sqrt((points[x[0]][0] - points[x[-1]][0])**2 + (points[x[0]][1] - points[x[-1]][1])**2)

    # return distance

class Evolution:
    chromosomes = []
    num_of_chromosomes = 0
    num_of_genes = 0
    elitism_chance = 0.0
    mutation_chance = 0.0

    def __init__(self, chromosomes, genes, mutation, elitism):
        self.num_of_genes = genes
        self.num_of_chromosomes = chromosomes
        self.mutation_chance = mutation
        self.elitism_chance = elitism

        for i in range(num_of_chromosomes):
            random.shuffle(filtered.pieces)
            self.chromosomes.append([piece.deep_copy() for piece in filtered.pieces])

    def roulette(self, num_of_chromosomes):
        fitness = [1/fitFun(x) for x in self.chromosomes]
        fitness = fitness / np.sum(fitness)
        new_chromosomes = []

        for i in range(num_of_chromosomes):
            num = random.random()
            sum = 0
            index = 0
            for n in fitness:
                sum += n
                if num < sum:
                    new_chromosomes.append(self.chromosomes[index])
                    break
                index += 1

        return new_chromosomes

    def crossover(self, chrom1, chrom2):
        a, b = random.sample(range(self.num_of_genes), 2)
        if a > b:
            a, b = b, a

        holes1, holes2 = [True] * self.num_of_genes, [True] * self.num_of_genes
        for i in range(self.num_of_genes):
            if i < a or i > b:
                holes1[chrom1[i]] = False
                holes2[chrom2[i]] = False

        temp1, temp2 = chrom1, chrom2
        k1, k2 = b + 1, b + 1
        for i in range(self.num_of_genes):
            if not holes1[temp1[(i + b + 1) % self.num_of_genes]]:
                chrom1[k1 % self.num_of_genes] = temp1[(i + b + 1) % self.num_of_genes]
                k1 += 1

            if not holes2[temp2[(i + b + 1) % self.num_of_genes]]:
                chrom2[k2 % self.num_of_genes] = temp2[(i + b + 1) % self.num_of_genes]
                k2 += 1

        for i in range(a, b + 1):
            chrom1[i], chrom2[i] = chrom2[i], chrom1[i]

        return chrom1, chrom2

    def mutation(self, chromosomes, probability):
        for chromosome in chromosomes:
            if random.random() < probability:
                index = random.randint(0, self.num_of_genes - 2)
                chromosome[index], chromosome[index+1] = chromosome[index+1], chromosome[index]

    def elitism(self):
        self.chromosomes.sort(key=fitFun, reverse = True)
        new_chromosomes = []
        for i in range(int(self.elitism_chance * self.num_of_chromosomes)):
            new_chromosomes.append(self.chromosomes[-i])
        return new_chromosomes

    def iteration(self):
        self.chromosomes.sort(key=fitFun, reverse = True)
        new_chromosomes = self.elitism()

        size_of_temp = int(self.num_of_chromosomes - self.elitism_chance * self.num_of_chromosomes)
        temp_population = self.roulette(size_of_temp)
        children = []

        for i in range(0, size_of_temp//2):

            child1, child2 = self.crossover([piece.deep_copy() for piece in temp_population[i]],[piece.deep_copy() for piece in temp_population[-i]])
            children.append(child1)
            children.append(child2)

        self.mutation(children, self.mutation_chance)
        new_chromosomes.extend(children)
        self.chromosomes = new_chromosomes
        self.chromosomes.sort(key=fitFun, reverse = True)



puzzle_collection = PuzzleCollection.unpickle()
filtered, _ = puzzle_collection.partition_by_notch_type(NotchType.NONE)

num_of_chromosomes = 100
num_of_genes = len(filtered.pieces)

# cities = [i for i in range(25)]

# points = [(119, 38), (37, 38), (197, 55), (85, 165), (12, 50), (100, 53), (81, 142), (121, 137), (85, 145),
#          (80, 197), (91, 176), (106, 55), (123, 57), (40, 81), (78, 125), (190, 46), (187, 40), (37, 107),
#          (17, 11), (67, 56), (78, 133), (87, 23), (184, 197), (111, 12), (66, 178)]


Ev = Evolution(num_of_chromosomes, num_of_genes, 0.01, 0.2)
for it in tqdm(range(1000)):
    Ev.iteration()

print(fitFun(Ev.chromosomes[-1]))