from skimage import io, color, measure, morphology, filters
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np
import math
from datetime import datetime
startTime = datetime.now()

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


def check_orientation(image):
    y = np.array([sum(row) for row in image if sum(row) > 0])
    transposed = image.transpose()
    x = np.array([sum(row) for row in transposed if sum(row) > 0])

    x = np.convolve(x, np.ones(10)/10, mode = 'valid')
    y = np.convolve(y, np.ones(10)/10, mode = 'valid')

    y =  y - y.min()
    y = y /  y.max()

    x = x - x.min()
    x = x / x.max()

    std = np.std(x) + np.std(y)

    if std <= 0.5:
        return 1
    else:
        return 0

def find_edges(mid_x, mid_y, x, y, thresh):

    min_x_upp = min(x[mid_x:])
    x_upp = [value-min_x_upp for value in x[mid_x:]]
    for idx, value in enumerate(x_upp):
        if value < thresh:
            x_edge_upp = idx+mid_x
            break

    min_x_low = min(x[mid_x::-1])
    x_low = [value - min_x_low for value in x[mid_x::-1]]
    for idx, value in enumerate(x_low):
        if value < thresh:
            x_edge_low = mid_x-idx
            break

    min_y_upp = min(y[mid_y:])
    y_upp = [value-min_y_upp for value in y[mid_y:]]
    for idx, value in enumerate(y_upp):
        if value < thresh:
            y_edge_upp = idx+mid_y
            break

    min_y_low = min(y[mid_y::-1])
    y_low = [value - min_y_low for value in y[mid_y::-1]]
    for idx, value in enumerate(y_low):
        if value < thresh:
            y_edge_low = mid_y-idx
            break

    return [x_edge_upp, x_edge_low], [y_edge_upp,  y_edge_low]


def edge_to_corner(x_edge, y_edge):
    corner_matrix = [[0,0], [0,1], [1,1], [1,0]]
    corner_coords = []
    for corner in corner_matrix:
        corner_coords.append([x_edge[corner[0]], y_edge[corner[1]]])

    return corner_coords


def crop_to_edge(image, x_edge, y_edge, mid_x, mid_y):
    image = image[y_edge[1]: y_edge[0],x_edge[1]:x_edge[0]]
    labels = morphology.label(image, connectivity=2)
    orientation = check_orientation(image)
    assert( labels.max() != 0 )
    image = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return image, mid_x - x_edge[1], mid_y - y_edge[1], orientation


def plot_corners(corner_coords):
    corner_ofset = [corner_coords[(idx+1)%4] for idx, _ in enumerate(corner_coords)]
    for corner1, corner2 in zip(corner_coords, corner_ofset):
        plt.plot([corner1[0], corner2[0]], [corner1[1], corner2[1]])


def get_x_y_pairs(image):
    _, colomns = np.shape(image)
    flat = image.flatten()
    x_y_pair = [[idx%colomns, idx//colomns]  for idx, value in enumerate(flat) if value == 1]
    return x_y_pair


def get_euclidean_distances(x_y_pairs, mid_x, mid_y):
    distances = [ math.sqrt((value[0] - mid_x)**2 + (value[1] - mid_y)**2) for value in x_y_pairs]
    return distances


def get_corners_tilted(x_y_pairs, distances, mid_x, mid_y):

    dist_upp = 20
    dist_low = 20
    dist_left = 20
    dist_right = 20

    for coord, dist in zip(x_y_pairs, distances):
        x = coord[0] - mid_x
        y = coord[1] - mid_y

        if x < y:
            if (x < -1*y) and (dist > dist_left):
                corn_left = coord
                dist_left = dist
            elif (x > -1*y) and (dist > dist_upp):
                corn_upp = coord
                dist_upp = dist
        elif x > y:
            if (x > -1*y) and (dist > dist_right):
                corn_right = coord
                dist_right = dist
            if (x < -1*y) and (dist > dist_low):
                corn_low = coord
                dist_low = dist
    
    return [corn_left, corn_right, corn_upp, corn_low]


def get_corners_straight(x_y_pairs, distances, mid_x, mid_y):
    dist_upp = 20
    dist_low = 20
    dist_left = 20
    dist_right = 20

    for coord, dist in zip(x_y_pairs, distances):
        x = coord[0] - mid_x
        y = coord[1] - mid_y

        if x < 0:
            if (y < 0) and (dist > dist_left):
                corn_left = coord
                dist_left = dist
            elif (y > 0) and (dist > dist_upp):
                corn_upp = coord
                dist_upp = dist
        elif x > 0:
            if (y > 0) and (dist > dist_right):
                corn_right = coord
                dist_right = dist
            if (y < 0) and (dist > dist_low):
                corn_low = coord
                dist_low = dist
    
    return [corn_left, corn_right, corn_upp, corn_low]


def find_rc(point1, point2):
    diff_x = point1[0] - point2[0]
    diff_y = point1[1] - point2[1]
    return diff_y/diff_x


def interpolate(image, point):
    lower_left = [ value // 1 for value in point]
    offsets = [value % 1 for value in point]
    cell = [[1,1], [1,0], [0,1], [0,0]]

    total_value = 0
    for corner in cell:
        weight = 1
        for offset, value in zip(offsets, corner):
            if value == 1:
                weight *= offset
            else:
                weight *= 1 - offset

        total_value += image[int(lower_left[0]+corner[0]),int(lower_left[1]+corner[1])]*weight
    return total_value


def translate_crop(corners, samples):
    rc_left = find_rc(corner[0], corner[2])
    rc_right = find_rc(corner[3], corner[1])

    x_diff_left = corner[2][0] - corner[0][0]
    x_diff_right = corner[1][0] - corner[3][0]

    step_left = x_diff_left/samples
    step_right = x_diff_right/samples


image_paths =  ['sudoku_straight.jpeg', 'sudoku_diag.jpeg',  'sudoku_straight_tilted.jpeg', 'sudoku_diag_tilted.jpeg']
plt.figure(1)

for idx, path in enumerate(image_paths):
    image = preprocess(path)

    dark = thresh(image, 0.6)
    edges = edges_binar(dark)

    mid_x, mid_y, x, y = find_mid(edges)
    x_edge, y_edge = find_edges(mid_x, mid_y, x, y,2)

    edges, mid_x_new, mid_y_new, orientation = crop_to_edge(edges, x_edge, y_edge, mid_x, mid_y)

    x_y_pairs = get_x_y_pairs(edges)
    distances = get_euclidean_distances(x_y_pairs, mid_x_new, mid_y_new)

    if orientation == 1:
        corners = get_corners_straight(x_y_pairs, distances, mid_x_new, mid_y_new)
    else:
        corners = get_corners_tilted(x_y_pairs, distances, mid_x_new, mid_y_new)

    print(datetime.now() - startTime)
    plt.subplot(2, 3, (idx+1))
    plt.imshow(image)
    for corner in corners:
        plt.plot(corner[0]+x_edge[1], corner[1]+y_edge[1], 'ro')

io.show()
