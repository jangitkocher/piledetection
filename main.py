import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from numpy import asarray
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float
import math

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2


def resise(source_path, target_path, width, height):
    try:
        img = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
        dim = (width, height)
        if (img.shape[1] * img.shape[0]) >= (width * height):
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(target_path, resized)
    except Exception as e:
        print(str(e))


def create_canny_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edge_detection_intensity = cv2.Canny(blurred, 300, 10)
    cv2.imwrite(path, edge_detection_intensity)


print("TensorFlow Version: ", tf.__version__)

np.random.seed(42)
tf.random.set_seed(42)

IMAGE_SIZE = 256
EPOCHS = 1000000
BATCH = 32
LR = 1e-4

PATH = "CVC-612/"


def model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")

    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output

    f = [16, 32, 48, 64]
    x = encoder_output
    for i in range(1, len(skip_connection_names) + 1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])

        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs, x)
    return model


def read_image_(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x = cv2.imread('/content/drive/MyDrive/DJI_01472.jpg', cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    return x


def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


def predict_pile(source_path):
    img = read_image_(source_path)
    y_pred = model.predict(np.expand_dims(img, axis=0))[0] > 0.5
    h, w, _ = img.shape
    white_line = np.ones((h, 10, 3))
    all_images = [white_line, mask_parse(y_pred)]
    image = np.concatenate(all_images, axis=1)
    plt.axis('off')
    imgplot = plt.imshow(image)
    plt.savefig(source_path, dpi=1000)


def get_img_width(path):
    img = cv2.imread(path)
    return img.shape[1]


def get_img_height(path):
    img = cv2.imread(path)
    return img.shape[0]


def add_black_background(path):
    img = Image.open(path)
    img_w, img_h = img.size
    background = Image.new('RGB', (img_w + int(img_w * 0.1), img_h + int(img_h * 0.1)), (0, 0, 0, 0))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    background.save(path)


def cut(source, target):
    img = Image.open(source).convert('RGBA')
    seed = (0, 0)
    rep_value = (0, 0, 0, 0)
    ImageDraw.floodfill(img, seed, rep_value, thresh=100)
    img.save(target)


def crop_only_once(path):
    image = cv2.imread(path)
    image_crop = image[580:4265, 1510:5195]
    cv2.imwrite(path, image_crop)


def complete_crop(original_image_path, predicted_pile_path):
    original_imge = cv2.imread(original_image_path)
    pred_mask_img = cv2.imread(predicted_pile_path)
    onepointzero_percent_width = int(original_imge.shape[1] * 0.01)
    onepointzero_percent_height = int(original_imge.shape[0] * 0.01)
    gray = cv2.cvtColor(pred_mask_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for (i, c) in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(c)
        cropped_contour = original_imge[y - onepointzero_percent_height:y + h + onepointzero_percent_height, x - onepointzero_percent_width:x + w + onepointzero_percent_width]
        # cropped_contour = original_imge[y:y + h, x:x + w]
        break
    cv2.imwrite(original_image_path, cropped_contour)


def baseline_coordinates(image):
    img = Image.open(image).convert('RGBA')
    w = get_img_width(image)
    h = get_img_height(image)
    coordinates = []
    for x in range(w):
        for y in range(h):
            current_color = img.getpixel( (x,y) )
            R,G,B,A = current_color
            if(R >= 200 and G >= 200 and B >= 200):
                coordinates.append((x, y))
                continue
    return coordinates


def right_bottom_point(array):
    # max = 0
    # for element in array:
    #     old = element[0]
    #     new = element[0]
    #     if(new >= old):
    #         max = element
    return array[-1]


def left_bottom_point(array):
    return array[0]


def to_array(image):
    image = Image.open(image)
    data = asarray(image)
    for i in data:
        print(data[i])
    smallest = np.amin(image)
    biggest = np.amax(image)
    print(smallest, biggest)
    new_image = Image.fromarray(data)
    new_image.save('array.jpg')


def calculate_angle(mask):
    coordinates = baseline_coordinates(mask)
    point1 = left_bottom_point(coordinates)
    point2 = right_bottom_point(coordinates)
    point3 = (point2[0], point1[1])

    lengthp1_p2 = abs(math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))
    lengthp2_p3 = abs(math.sqrt((point3[0] - point2[0]) ** 2 + (point3[1] - point2[1]) ** 2))
    lengthp3_p1 = abs(math.sqrt((point1[0] - point3[0]) ** 2 + (point1[1] - point3[1]) ** 2))

    angle = (180 / math.pi) * (math.acos((abs(lengthp2_p3 ** 2 - lengthp1_p2 ** 2 - lengthp3_p1 ** 2)) / (2 * lengthp1_p2 * lengthp3_p1)))

    if (point2[1] >= point1[1]):
        return angle
    else:
        return angle * (-1)


def rotate_image(path, angle):
    image = Image.open(path)
    rotated = image.rotate(angle)
    rotated.save(path)


def resizePil(source, target, x, y):
    image = Image.open(source)
    image = image.resize((x, y), Image.ANTIALIAS)
    image.save(target, quality=100)


def blend_images(img1, img2):
    w = get_img_width(img1)
    h = get_img_height(img1)
    resizePil(img2, img2, w, h)
    img1open = Image.open(img1).convert('RGBA')
    img2open = Image.open(img2)
    result = Image.blend(img1open, img2open, alpha=0.4).convert('RGB')
    result.save('blend.jpg')


def change_color(image):
    img = Image.open(image).convert('RGBA')
    # r, g, b = img.getpixel( (0,0) )
    w = get_img_width(image)
    h = get_img_height(image)
    for x in range(w):
        for y in range(h):
            current_color = img.getpixel( (x,y) )
            R,G,B,A = current_color
            if(R >= 55 and G >= 55 and B >= 55):
                new_color = (0,255,0)
                img.putpixel( (x,y), new_color)
            # else:
            #     new_color = (105,105,105)
            #     img.putpixel( (x,y), new_color)
    img.save(image)


def transparent(image):
    img = Image.open(image).convert('RGBA')
    pixdata = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (124,252,0, 255):
                pixdata[x, y] = (124,252,0, 100)
    img.save(image, 'PNG')


def merge_images(img1, img2):
    w = get_img_width(img1)
    h = get_img_height(img1)
    resizePil(img2, img2, w, h)

    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)
    result = cv2.addWeighted(image1, 1, image2, 0.5, 0)
    # image1 = Image.open(img1).convert('RGBA')
    # image2 = Image.open(img2)
    # image2.paste(image1, (0,0))
    # image2.save('merged.png', 'PNG')
    cv2.imwrite('merged.png', result)


if __name__ == '__main__':
    ##################################
    source_image_path = 'DJI_0138.jpg'
    ##################################

    target_image_path = source_image_path.replace('.jpg', '_pred') + '.jpg'

    saved_model_path = 'piledetection.h5'

    source_copy = cv2.imread(source_image_path)
    cv2.imwrite('source_copy.jpg', source_copy)

    resise(source_image_path, target_image_path, 512, 512)
    create_canny_image(target_image_path)

    model = model()
    model.summary()
    model.load_weights(saved_model_path)

    predict_pile(target_image_path)

    crop_only_once(target_image_path)

    resise(target_image_path, target_image_path, get_img_width(source_image_path), get_img_height(source_image_path))

    add_black_background(source_image_path)
    add_black_background(target_image_path)

    target_copy = cv2.imread(target_image_path)
    cv2.imwrite('target_copy.jpg', target_copy)

    complete_crop(source_image_path, target_image_path)


    complete_crop(target_image_path, target_image_path)

    rotate_image(source_image_path, calculate_angle(target_image_path))
    rotate_image(target_image_path, calculate_angle(target_image_path))

    target_image_path_png = target_image_path.replace('.jpg', '.png')

    cut(target_image_path, target_image_path_png)
    change_color(target_image_path_png)
    transparent(target_image_path_png)

    merge_images(source_image_path, target_image_path_png)
