import cv2
import numpy as np

def save_image(path, image):
    cv2.imwrite(path, image)

def threshold(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return binary_image

def view_image(image):
    image_shape = image.shape
    print("viewing image")
    #fit image to screen
    if image.shape[0] >= 1080 or image.shape[1] >= 1920:
        max_size = max(image.shape[0], image.shape[1])
        scale = 1080 / max_size
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, new_size)
    cv2.imshow(f"shape = {image_shape}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def enlarge_image(image, scale):
    height, width = image.shape[:2]
    new_width = int(width*scale)
    new_height = int(height*scale)
    if len(image.shape) == 3:
        new_image = np.zeros((new_height,new_width,image.shape[2]), dtype=np.uint8)
    else:
        new_image = np.zeros((new_height,new_width), dtype=np.uint8)

    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2

    new_image[start_y:start_y+height, start_x:start_x+width] = image

    return new_image

def enlarge_image_binary(image, scale):
    height, width = image.shape[:2]
    new_width = int(width*scale)
    new_height = int(height*scale)

    new_image = np.zeros((new_height,new_width), dtype=np.uint8)

    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2

    new_image[start_y:start_y+height, start_x:start_x+width] = image

    return new_image

def load_image(filename):
    print("loading image from file: ", filename)
    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"File {filename} not found")
    return image

def scroll_image(knob_image, vector):
    x, y = vector
    height, width = knob_image.shape[:2]
    new_image = np.zeros_like(knob_image)
    new_image[max(0, -x):min(height, height - x), max(0, -y):min(width, width - y)] = knob_image[max(0, x):min(height, height + x), max(0, y):min(width, width + y)]
    return new_image