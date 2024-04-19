import unittest
import genetic_algorithm
from teeth_detection import NotchType


class MyTestCase(unittest.TestCase):
    def test_edges_to_test(self):
        result = genetic_algorithm.edges_to_test(
            {"TOP": NotchType.NONE, "RIGHT": NotchType.TOOTH, "BOTTOM": NotchType.HOLE, "LEFT": NotchType.HOLE})
        self.assertEqual(result, ("RIGHT", "LEFT"))

        result = genetic_algorithm.edges_to_test(
            {"TOP": NotchType.TOOTH, "RIGHT": NotchType.HOLE, "BOTTOM": NotchType.NONE, "LEFT": NotchType.NONE})
        self.assertEqual(result, ("TOP", "BOTTOM"))

        result = genetic_algorithm.edges_to_test(
            {"TOP": NotchType.NONE, "RIGHT": NotchType.TOOTH, "BOTTOM": NotchType.HOLE, "LEFT": NotchType.NONE})
        self.assertEqual(result, ("RIGHT", "LEFT"))

        result = genetic_algorithm.edges_to_test(
            {"TOP": NotchType.NONE, "RIGHT": NotchType.NONE, "BOTTOM": NotchType.TOOTH, "LEFT": NotchType.HOLE})
        self.assertEqual(result, ("BOTTOM", "TOP"))



if __name__ == '__main__':
    unittest.main()
