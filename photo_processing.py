import cv2
import numpy as np

import image_processing


def process_photo(path= "input_photos/good_one.png"):
    SCALE = 0.25

    original = image_processing.load_image(path)

    new_shape = (int(original.shape[1] * SCALE), int(original.shape[0] * SCALE))
    resized = cv2.resize(original, new_shape)
    blurred = cv2.medianBlur(resized, 5)

    grayscale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    contours = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    result = np.zeros((new_shape[1], new_shape[0]), dtype=np.uint8)

    for contour in contours:
        cv2.drawContours(result, [contour], 0, 255, -1)

    final_gray = cv2.morphologyEx(result, cv2.MORPH_ERODE, kernel, iterations=2)

    final_rgb = cv2.bitwise_and(resized, resized, mask=final_gray)

    return final_rgb, final_gray


if __name__ == '__main__':
    #path = "input_photos/good_one.png"
    path = "puzzle_images/IMG_20240327_175301.jpg"
    image, mask = process_photo(path)

    image_processing.save_image("results/processed.png", image)
    image_processing.save_image("results/processed_mask.png", mask)
    image_processing.view_image(image)















