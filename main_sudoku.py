from skimage import io, color, measure, morphology
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np

def preprocess(image_path):
    img = io.imread(image_path)
    img = color.rgb2gray(img)
    img = img - img.min()
    img = img/img.max()
    return img


image = preprocess('sudoku_straight_tilted.jpeg')
lbp = local_binary_pattern(image, 8, 1, 'uniform')

test = [[1 if ele in [4] else 0 for ele in row]for row in lbp]
test = np.array(test)

se = morphology.disk(2)

test = morphology.dilation(test, selem=se)


io.imshow(test)
io.show()