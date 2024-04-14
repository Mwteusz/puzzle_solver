import cv2
import numpy as np

def save_image(path, image):
    cv2.imwrite(path, image)

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


def square(image):
    height, width = image.shape[:2]
    a = max(height, width)
    new_image = np.zeros((a, a, 3), dtype=np.uint8)
    new_image[:height, :width] = image
    return new_image



def put_text(image, text, point, color=(255,255,255), width=1, font=cv2.FONT_HERSHEY_SIMPLEX, param2=0.5, aa=cv2.LINE_AA):
    text_width, text_height = cv2.getTextSize(text, font, param2, width)[0]
    point = (point[0] - text_width // 2, point[1] + text_height // 2)
    cv2.putText(image, text, point, font, param2, color, width, aa)


def bound_image(edges):
    x, y, w, h = cv2.boundingRect(edges)
    bound = edges[y:y + h, x:x + w]
    return bound, (x,y)




def images_to_image(images):
    images = [cv2.resize(square(image), (200, 200)) for image in images]
    size = int(np.ceil(np.sqrt(len(images))))
    image_size = images[0].shape[0]
    image_array = np.zeros((image_size*size, image_size*size,3), dtype=np.uint8)
    for i, image in enumerate(images):
        x = i % size
        y = i // size
        start_x = x * image_size
        start_y = y * image_size
        image_array[start_y:start_y+image_size, start_x:start_x+image_size] = image
    return image_array