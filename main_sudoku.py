from skimage import io, color, measure, morphology, filters
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np


def preprocess(image_path):
    img = io.imread(image_path)
    img = color.rgb2gray(img)
    img = img - img.min()
    img = img/img.max()
    return img


def lbp_edge(image):
    lbp = local_binary_pattern(image, 8, 1, 'uniform')
    lbp_edge = lbp == 4
    lbp_edge = np.array(lbp_edge*1)
    return lbp_edge


def thresh(image, trh):
    treshed = image < trh
    treshed = np.array(treshed*1)
    return treshed


def find_patches(image, bins):
    patches = filters.gaussian(image, 5)
    range_img = patches.max()-patches.min()
    min_img = patches.min()
    patch_work = np.zeros(np.shape(patches))
    for i in range(1,bins+1):
        thresh = ((i/bins) * range_img) + min_img
        patch = patches < thresh
        patch_work += (patch*1)
    
    return patch_work


def edges_binar(image):
    strel  = morphology.disk(1)
    dilate = morphology.dilation(image, strel)
    edges = dilate-image
    return edges


def find_mid(image):
    y = [sum(row)*idx for idx, row in enumerate(image)]
    transposed = image.transpose()
    x = [sum(row)*idx for idx, row in enumerate(transposed)]
    total = sum(sum(image))
    return sum(x)//total, sum(y)//total, [sum(row) for idx, row in enumerate(transposed)], [sum(row) for idx, row in enumerate(image)]


def find_edges(mid_x, mid_y, x, y, thresh):

    for idx, value in enumerate(x[mid_x:]):
        if value < thresh:
            x_edge_upp = idx+mid_x
            break

    for idx, value in enumerate(x[mid_x::-1]):
        if value < thresh:
            x_edge_low = mid_x-idx
            break

    for idx, value in enumerate(y[mid_y:]):
        if value < thresh:
            y_edge_upp = idx+mid_y
            break

    for idx, value in enumerate(y[mid_y::-1]):
        if value < thresh:
            y_edge_low = mid_y-idx
            break

    return [x_edge_upp, x_edge_low], [y_edge_upp,  y_edge_low]


image_paths =  ['sudoku_straight.jpeg', 'sudoku_diag.jpeg',  'sudoku_straight_tilted.jpeg', 'sudoku_diag_tilted.jpeg']
plt.figure()
for idx, path in enumerate(image_paths):
    image = preprocess(path)

    dark = thresh(image, 0.6)
    edges = edges_binar(dark)
    mid_x, mid_y, x, y = find_mid(edges)
    x_edge, y_edge = find_edges(mid_x, mid_y, x, y,5)
    plt.subplot(2,3,(idx+1))
    plt.imshow(edges)
    plt.plot(x_edge, [y_edge[0], y_edge[0]])
    plt.plot([x_edge[0], x_edge[0]], y_edge)

io.show()