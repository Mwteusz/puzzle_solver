import unittest
import genetic_algorithm
import image_processing
from puzzle_extracting import PuzzleCollection
from teeth_detection import NotchType


class GeneticAlgorithmTestCase(unittest.TestCase):
    def test_edges_to_test(self):
        result = genetic_algorithm.edges_to_test(
            {"TOP": NotchType.NONE, "RIGHT": NotchType.TOOTH, "BOTTOM": NotchType.HOLE, "LEFT": NotchType.HOLE})
        self.assertEqual(result, ("RIGHT", "LEFT"))

        result = genetic_algorithm.edges_to_test(
            {"TOP": NotchType.TOOTH, "RIGHT": NotchType.HOLE, "BOTTOM": NotchType.NONE, "LEFT": NotchType.NONE})
        self.assertEqual(result, ("TOP", "BOTTOM"))

        result = genetic_algorithm.edges_to_test(
            {"TOP": NotchType.NONE, "RIGHT": NotchType.NONE, "BOTTOM": NotchType.TOOTH, "LEFT": NotchType.HOLE})
        self.assertEqual(result, ("BOTTOM", "TOP"))

    def test_fitFun(self):
        pass
        puzzle_collection = PuzzleCollection.unpickle(name="2024-04-14.pickle")
        filtered, _ = puzzle_collection.partition_by_notch_type(NotchType.NONE)
        filtered = filtered.pieces
        image_processing.view_image(puzzle_collection.get_preview())
        for i, piece in enumerate(filtered):
            piece.id = i

        evolution = genetic_algorithm.Evolution(100, len(filtered), 0.0, 0.1, 0.2)
        for chromosome in evolution.chromosomes:
            result = genetic_algorithm.fitFun(chromosome)
            self.assertAlmostEquals(result, 0, 2)

    def test_crossover(self):
        pass


if __name__ == '__main__':
    unittest.main()
