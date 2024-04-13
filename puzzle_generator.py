import math
import random

import numpy as np
import cv2
import scatter
import image_processing




def get_knob_from_neighbour(neighbour_image, circle_coords, radius):
    x, y = circle_coords
    mask = np.zeros_like(neighbour_image)
    cv2.circle(mask, (x, y), radius, (255, 255, 255), -1)
    cv2.rotate(mask, cv2.ROTATE_180, mask)
    image = cv2.bitwise_and(neighbour_image, mask)

    return image

def draw_knob(piece, neighbour, circle_coords, radius:int,vector):
    knob_image = get_knob_from_neighbour(neighbour.original, circle_coords, radius)
    knob_image = image_processing.scroll_image(knob_image, (vector[1]*-piece.width, vector[0]*-piece.height))

    #print(piece.original.shape, knob_image.shape)
    piece_with_knob = cv2.bitwise_or(piece.puzzle_image, knob_image)
    return piece_with_knob


def make_hole(image, circle_coords, radius:int):
    x, y = circle_coords
    piece_with_hole = image.copy()
    cv2.circle(piece_with_hole, (x, y), radius, (0, 0, 0), -1)
    return piece_with_hole
class Piece:


    def __init__(self, image, x, y, width, height):
        self.original = image_processing.enlarge_image(image, 1.5)
        self.puzzle_image = self.original.copy()
        self.x, self.y = x, y
        self.width, self.height = width, height
    def save_puzzle_image(self,path):
        print("saving image to file: ", path)
        image_processing.save_image(path, self.puzzle_image)


    def carve_knob(self, edge):
        node1, node2 = edge.node1, edge.node2

        if node1.piece.x == self.x and node1.piece.y == self.y:
            neighbour_node = node2
        else:
            neighbour_node = node1

        vector = (neighbour_node.piece.x - self.x, neighbour_node.piece.y - self.y)

        a = self.width
        radius:int = int(((a * math.sqrt(2)) / 4) * 0.6)

        image_center = (self.original.shape[0] // 2, self.original.shape[1] // 2)
        circle_center = vector[0] * (a//2), vector[1] * (a//2) #coords of the knob

        cicle_coords = (image_center[0] + circle_center[0], image_center[1] + circle_center[1])
        if self.equals(edge.piece_with_hole.piece):
            self.puzzle_image = make_hole(self.puzzle_image, cicle_coords, radius)
        else:
            self.puzzle_image = draw_knob(self, neighbour_node.piece, cicle_coords, radius,vector)
    def equals(self, piece):
        return self.x == piece.x and self.y == piece.y


class Node:


    def __init__(self, piece):
        self.piece = piece
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)
    def __str__(self):
        return f"[Node: {self.piece.x}, {self.piece.y}, {len(self.edges)} edges]"

class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.piece_with_hole = None

    def equals(self, edge):
        return (self.node1 == edge.node1 and self.node2 == edge.node2) or (self.node1 == edge.node2 and self.node2 == edge.node1)

    def set_hole(self, piece):
        self.piece_with_hole = piece
    def get_nodes(self):
        return [self.node1, self.node2]


#split into SQUARE images




def square_image(image):
    x, y, _ = image.shape
    a = min(x, y)
    return image[:a, :a]


def image_to_pieces(image, puzzle_size) -> np.ndarray:
    #check if image can be split into squares
    #if image.shape[0] % grid_size_y != 0 or image.shape[1] % grid_size_x != 0:
    #print("cutting off edges")
    #image = square_image(image)

    height, width, _ = image.shape
    grid_size_x = width // puzzle_size
    grid_size_y = height // puzzle_size

    puzzle_grid = np.zeros((grid_size_x, grid_size_y), dtype=object)

    for x in range(grid_size_x):
        for y in range(grid_size_y):
            start_x = x * puzzle_size
            start_y = y * puzzle_size
            end_x = start_x + puzzle_size
            end_y = start_y + puzzle_size

            piece_image = image[start_y:end_y, start_x:end_x]
            #view_image(piece_image)
            piece = Piece(piece_image, x, y, puzzle_size, puzzle_size)
            puzzle_grid[x, y] = piece

    print(f"processed the image into {puzzle_grid.size} pieces ({grid_size_x}x{grid_size_y})")
    return puzzle_grid




def find_edge(all_edges, new_edge):
    for edge in all_edges:
        if new_edge.equals(edge):
            return edge
    return None





def create_edges(nodes, grid_size_x, grid_size_y):

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    all_edges = []
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            for direction in directions:
                new_x = x + direction[0]
                new_y = y + direction[1]

                if new_x >= 0 and new_x < grid_size_x and new_y >= 0 and new_y < grid_size_y:
                    new_edge = Edge(nodes[x, y], nodes[new_x, new_y])
                    edge = find_edge(all_edges, new_edge)

                    if edge is None:
                        all_edges.append(new_edge)
                        edge = new_edge
                    nodes[x, y].add_edge(edge)
    return all_edges


def create_nodes(puzzle_grid, grid_size_x, grid_size_y):
    nodes = np.zeros((grid_size_x, grid_size_y), dtype=object)
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            nodes[x, y] = Node(puzzle_grid[x, y])
    return nodes


def carve_knobs(all_edges):
    for edge in all_edges:
        edge_nodes = edge.get_nodes() #neighbouring nodes (pieces)

        random_piece = random.choice(edge_nodes)
        edge.set_hole(random_piece)

        for node in edge_nodes:
            node.piece.carve_knob(edge)


def save_puzzles(nodes, grid_size_x, grid_size_y):
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            nodes[x, y].piece.save_puzzle_image(f"pieces/puzzle_piece_{x}_{y}.png")


def get_pieces_from_graph(nodes)->list:
    pieces = []
    for x in range(nodes.shape[0]):
        for y in range(nodes.shape[1]):
            pieces.append(nodes[x, y].piece)
    return pieces

def replace_color(image, old_color, new_color):
    result = image.copy()
    result[np.where((result == old_color).all(axis=2))] = new_color
    return result


def remove_zeros_from_rgb(image):
    new_colors = []
    for color in cv2.split(image):
        new_color = color.copy()
        new_color[np.where(new_color == 0)] = 1
        new_colors.append(new_color)
    return cv2.merge(new_colors)


def create_puzzles(image, puzzle_size):
    image = remove_zeros_from_rgb(image)
    puzzle_grid = image_to_pieces(image, puzzle_size)
    grid_size_x, grid_size_y = puzzle_grid.shape
    print("grid size: ",grid_size_x, grid_size_y)

    nodes = create_nodes(puzzle_grid, grid_size_x, grid_size_y)
    all_edges = create_edges(nodes, grid_size_x, grid_size_y)
    carve_knobs(all_edges)
    pieces = get_pieces_from_graph(nodes)
    #save_puzzles(nodes, grid_size_x, grid_size_y)
    return pieces, (grid_size_x, grid_size_y)




def image_to_puzzles(path = "input_photos/bliss.png", vertical_puzzle_size = 5, image = None):
    if image is None:
        image = image_processing.load_image(path)
    puzzle_size = image.shape[0] // vertical_puzzle_size
    if puzzle_size <= 70:
        raise ValueError(f"Puzzles sized at {puzzle_size}x{puzzle_size} are too small. Decrease the vertical_puzzle_size parameter or scale up the image.")

    pieces, grid = create_puzzles(image, puzzle_size)


    output_size = (image.shape[0]*2, image.shape[1]*2)

    scattered_puzzle = scatter.scatter_pieces(output_size, pieces=pieces, minimum_distance=10)
    scattered_puzzle_mask = image_processing.threshold(scattered_puzzle,0)

    return scattered_puzzle, scattered_puzzle_mask




if __name__ == '__main__':

    image_names = ["bliss", "coolimage","dom","dywan","good_one","gorawino2","lake"]
    for name in image_names:
        path = f"input_photos/{name}.png"
        scattered_puzzle, mask = image_to_puzzles(path, 3)

        image_processing.save_image(f"results/{name}.png", scattered_puzzle)
        image_processing.save_image(f"results/{name}_mask.png", mask)
        #image_processing.view_image(scattered_puzzle)










