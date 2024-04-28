import cv2
import numpy as np
import os
import imageio

def save_image(path, image):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path, image)

def save_gif(path, frames):
    """:param frames: list of images"""
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    imageio.mimsave(path, frames, fps=2)

def threshold(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return binary_image

def closest_multiple(lower_bound, x):
    m = 1
    while x < lower_bound:
        x += lower_bound
        m += 1
    return m

lower_bound, upper_bound = 200, 1000
def view_image(image, title=None, fit_to_screen=True):
    if image is None:
        raise ValueError("image is None")
    final_title = f"s={image.shape[:2]}"
    if fit_to_screen:
        if image.shape[0] >= upper_bound or image.shape[1] >= upper_bound:
            scale = upper_bound / max(image.shape[0], image.shape[1])
            new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
            image = cv2.resize(image, new_size)
            final_title += f" ({scale:.2f}x scale)"
        elif image.shape[0] <= lower_bound or image.shape[1] <= lower_bound:
            min_size = min(image.shape[0], image.shape[1])
            max_size = max(image.shape[0], image.shape[1])
            scale = closest_multiple(lower_bound, min_size)
            a, b = min_size * scale, max_size * scale
            new_size = (a,b) if image.shape[0] > image.shape[1] else (b,a)
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
            final_title += f" ({scale}x scale)"

    if title is not None:
        final_title = f"[{title}] {final_title}"
    cv2.imshow(final_title, image)
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


def fit_square(image):
    """Fits the image into a square, by adding black padding."""
    height, width = image.shape[:2]
    a = max(height, width)
    if len(image.shape) == 2:
        new_image = np.zeros((a, a), dtype=np.uint8)
    else:
        new_image = np.zeros((a, a, 3), dtype=np.uint8)
    start_x = (a - width) // 2
    start_y = (a - height) // 2
    new_image[start_y:start_y + height, start_x:start_x + width] = image
    return new_image

def crop_square(image, a):
    """Crops the image to a square of size a. The center of the image is used as the center of the square."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    result =  image[int(center[1]-a//2):int(center[1]+a//2), int(center[0]-a//2):int(center[0]+a//2)]
    new_a = result.shape[0]
    return result, new_a
#def crop_square(image):
#    height, width = image.shape[:2]
#    sml, big = min(height, width), max(height, width)
#    start = (big - sml) // 2
#    end = start + sml
#    if sml == height:
#        result = image[:, start:end]
#    else:
#        result = image[start:end, :]
#    return result



def put_text(image, text, point, color=(255,255,255), width=1, font=cv2.FONT_HERSHEY_SIMPLEX, param2=0.5, aa=cv2.LINE_AA):
    text_width, text_height = cv2.getTextSize(text, font, param2, width)[0]
    point = (point[0] - text_width // 2, point[1] + text_height // 2)
    cv2.putText(image, text, point, font, param2, color, width, aa)


def bound_image(image):
    x, y, w, h = cv2.boundingRect(image)
    bound = image[y:y + h, x:x + w]
    return bound, (x,y)



def turn_binary_to_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
def images_to_image(images, interpolation=cv2.INTER_NEAREST):
    images = [cv2.resize(turn_binary_to_rgb(fit_square(image)), (200, 200), interpolation=interpolation) for image in images]
    size = int(np.ceil(np.sqrt(len(images))))
    image_size = images[0].shape[0]
    if len(images[0].shape) == 3:
        image_array = np.zeros((image_size*size, image_size*size, 3), dtype=np.uint8)
    else:
        image_array = np.zeros((image_size * size, image_size * size), dtype=np.uint8)
    for i, image in enumerate(images):
        x = i % size
        y = i // size
        start_x = x * image_size
        start_y = y * image_size
        image_array[start_y:start_y+image_size, start_x:start_x+image_size] = image
    return image_array


def erode(mask, i):
    kernel = np.ones((i, i), np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def get_cross(a,thickness=None):
    thickness = int(a*0.33) if thickness is None else thickness
    padding = int((a-thickness)//2)
    """:returns a plus sign of size a"""
    result = np.zeros((a, a), dtype=np.uint8)
    start = padding
    end = a - padding
    result[start:end, :] = 255
    result[:, start:end] = 255
    return result


def get_circle(a,r):
    return cv2.circle(np.zeros((a,a), dtype=np.uint8), (a//2, a//2), r, 255, -1)

def get_rhombus(a):
    result = np.zeros((a, a), dtype=np.uint8)
    corners =[ (a//2, 0), (a-1, a//2), (a//2, a-1), (0, a//2)]

    cv2.fillPoly(result, [np.array(corners)], 255)
    return result


def add_border(image, padding, color=0):

    a = image.shape[0]
    b = image.shape[1]
    image[0:padding, :] = color
    image[:, 0:padding] = color
    image[a-padding:, :] = color
    image[:,b-padding:] = color


def resize_image(image, scale=None, interpolation=None, size=None):
    if scale is not None:
        return cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=interpolation)
    if size is not None:
        return cv2.resize(image, size, interpolation=interpolation)


def expand_right_bottom(image, max_height, max_width):
    new_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    new_image[:image.shape[0], :image.shape[1]] = image
    return new_image

def draw_arrow(image, start, end,color=(255,255,255), thickness=2, type="arrow"):
    if type == "arrow":
        image = cv2.arrowedLine(image, start, end, color, thickness)
    elif type == "line":
        image = cv2.line(image, start, end, color, thickness)
    return image