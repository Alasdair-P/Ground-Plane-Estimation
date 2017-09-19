from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import cv2
import time
import os
from MSDC import main_ft.py
from scipy import misc


def ransac_plane_with_mask(data, input_mask=None, no_iter=None, threshold=None):
    '''
    Robustly fits a plane through a a cloud of points, using a mask to select chosen points

    INPUTS
    data - matrix of data points, n by 3, each line is (x1, y1, z1)
    no_iter - number of interations allowed in the algorithm
    threshold - number of std deviations for determining when a data point fits the model
    input_mask - binary mask of which pixels are valid for selection

    OUTPUTS
    V - vector of the unit normal of the plane ()
    C - the distance between the plane and the normal
    output_mask - binary mask of which pixels were used in mask
    '''

    points = np.array(data)

    assert points.shape[1] == 3
    assert points.shape[0] >= 4
    assert no_iter >= 1
    assert threshold >= 0

    no_points = points.shape[0]

    if input_mask is None:
        input_mask = np.ones((no_points, 1))

    if no_iter is None:
        no_iter = 1000

    if threshold is None:
        std = np.std(points)
        threshold = std * 1e-4

    v_bestfit = np.array([0, 0, 0])
    c_bestfit = np.array([0])
    best_no_inliers = 0
    output_mask = input_mask

    for iter in range(no_iter):  # loops over interations

        for i in range(100):

            test_points = points[np.random.permutation(no_points)[:3], :]  # select three random points form the data

            vec_1 = test_points[0, :] - test_points[1, :]  # select two vectors that describe this plane
            vec_2 = test_points[0, :] - test_points[2, :]

            v = np.cross(vec_1, vec_2) / np.linalg.norm(np.cross(vec_1, vec_2))  # calculate normal of the plane

            c = np.dot(test_points[0, :].reshape(3), v.reshape(3))

            if not (v[0] == 0 and v[1] == 0 and v[2] == 0):  # check for collinearity
                break

        inleirs = np.absolute(np.matmul(points, v.reshape(3, 1)) - c) < threshold

        inleirs_mask = inleirs.astype(int) * input_mask.astype(int)

        no_inliers = sum(inleirs_mask) - 3

        if no_inliers > best_no_inliers:
            best_no_inliers = no_inliers
            output_mask = inleirs
            # v_bestfit = v
            # c_bestfit = c
            print('best_no_inliers: ', best_no_inliers, '| Total possilbe inliers: ', no_points, '| Iteration: ', iter)

    # OLS regression to find best plane, for inlying points

    inleirs = ((output_mask.astype(int) * input_mask.astype(int))>0.5).reshape(-1)

    # inleirs = output_mask.reshape(-1)

    X = np.hstack((np.ones((no_points, 1)),(data[:,0:2])))
    X = X[inleirs, :]

    Y = data[:,2]
    Y = Y[inleirs]

    Beta = np.matmul( np.linalg.pinv((np.matmul(X.T, X) + np.eye(3) * 1e-6)) , np.matmul(X.T, Y))

    # convert to normal form

    d = -Beta[0]
    a = Beta[1]
    b = Beta[2]
    c = -1

    denominator = np.sqrt(np.square(a) + np.square(b) + np.square(c))

    v_bestfit = np.array((a, b, c)).T / denominator

    c_bestfit = d / denominator

    if v_bestfit[1]<0:
	v_bestfit = -v_bestfit
	c_bestfit = -c_bestfit

    print('V: ', v_bestfit, 'C: ', c_bestfit)

    return v_bestfit, c_bestfit, output_mask


def gray2rgb(im, cmap='plasma'):
    # taken from https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgba_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap='plasma'):
    # taken from https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    depth = depth
    return depth


def occlusion_mask(data, v, c, threshold):
    '''
    INPUTS
    data - matrix of data points, n by 3, each line is (x1, y1, z1)
    V - vector of the unit normal of the plane ()
    C - the distance between the plane and the normal

    OUTPUTS
    output_mask - binary mask of which pixels were used in mask
    '''
    points = np.array(data)

    if threshold is None:
        std = np.std(points)
        threshold = std * 1e-4

    if v[1] < 0:
        v = -v
        c = -c
        print('flipped!')

    output_mask = np.matmul(points, v.reshape(3, 1)) - c - threshold < 0

    return output_mask


def dispartiy_to_world_points(disparity, K=None, b=None, focal_l=None):
    '''
    INPUTS
    disparity - dense disparity map
    K - Intersic matrix of the camera
    b - baseline between cameras for trained model

    OUTPUTS
    world_points - matrix of world points, each line is (x1, y1, z1)

    '''

    height_pixels = disparity.shape[0]  # 256
    width_pixels = disparity.shape[1]  # 512

    disparity = disparity * disparity.shape[1]

    if b is None:
        b = 0.16

    if K is None:
        h_ds = 1920 / width_pixels  # horizontal downs ample 3.75
        v_ds = 1080 / height_pixels  # vertical downs ample 4.21875

        K = np.array([[1.40597825e+03 / h_ds, 0.00000000e+00, 9.69946177e+02 / h_ds],
                      [0.00000000e+00, 1.40675654e+03 / v_ds, 5.58178743e+02 / v_ds],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    if focal_l is None:
        focal_l = K[0, 0]

    height_pixels = disparity.shape[0]  # 256
    width_pixels = disparity.shape[1]  # 512

    x_im = (np.arange(width_pixels).reshape(1, -1) * np.ones((height_pixels, 1)))
    y_im = np.flipud(np.arange(height_pixels).reshape(-1, 1) * np.ones((1, width_pixels)))

    z_world = (focal_l / disparity) * b
    x_world = z_world / (K[0, 0]) * (x_im - K[0, 2])
    y_world = z_world / (K[1, 1]) * (y_im - K[1, 2])

    Data = np.concatenate((x_world.reshape(-1, 1), y_world.reshape(-1, 1), z_world.reshape(-1, 1)), axis=1)

    return Data, z_world, y_world, x_world


def image_to_world_points(data, K=None, b=None, focal_l=None):
    '''
    INPUTS
    disparity - dense disparity map
    K - Intersic matrix of the camera
    b - baseline between cameras for trained model

    OUTPUTS
    world_points - matrix of world points, each line is (x1, y1, z1)

    '''

    if K is None:
        h_ds = 1920 / 512  # horizontal downs ample 3.75
        v_ds = 1080 / 256  # vertical downs ample 4.21875

        K = np.array([[1.40597825e+03 / h_ds, 0.00000000e+00, 9.69946177e+02 / h_ds],
                      [0.00000000e+00, 1.40675654e+03 / v_ds, 5.58178743e+02 / v_ds],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    x_im = data[:,0]
    y_im = data[:,1]

    # K[0, 2] = 256
    # K[1, 2] = 128

    z_world = data[:,2]
    x_world = z_world / (K[0, 0]) * (x_im - K[0, 2])
    y_world = z_world / (K[1, 1]) * (y_im - K[1, 2])

    Data = np.hstack((x_world.reshape(-1, 1), y_world.reshape(-1, 1), z_world.reshape(-1, 1)))

    return Data


def gen_grid_points(grid_size_x=None, grid_size_y=None):
    '''
    INPUTS
    disparity - dense disparity map
    K - Intersic matrix of the camera
    b - baseline between cameras for trained model

    OUTPUTS
    world_points - matrix of world points, each line is (x1, y1, z1)

    '''
    if grid_size_x is None:
        grid_size_x = 10
    if grid_size_y is None:
        grid_size_y = 20

    print('grid_size_x', grid_size_x, 'grid_size_y', grid_size_y)
    x_world = np.arange(-(grid_size_x - 1) / 2, (grid_size_x + 1) / 2).reshape(1, -1) * np.ones((grid_size_y, 1))
    y_world = np.arange(-(grid_size_y - 1) / 2, (grid_size_y + 1) / 2).reshape(-1, 1) * np.ones((1, grid_size_x))
    z_world = np.zeros((grid_size_x, grid_size_y))

    plane_points = np.concatenate((x_world.reshape(-1, 1), y_world.reshape(-1, 1), z_world.reshape(-1, 1)), axis=1)
    return plane_points


def gen_edge_points(Z, Y, scale=None, grid_size_x=None, grid_size_y=None):
    '''
    INPUTS
    disparity - dense disparity map
    K - Intersic matrix of the camera
    b - baseline between cameras for trained model

    OUTPUTS
    world_points - matrix of world points, each line is (x1, y1, z1)

    '''
    if grid_size_x is None:
        grid_size_x = 10
    if grid_size_y is None:
        grid_size_y = 10
    if scale is None:
        scale = 1

    # print('grid_size_x', grid_size_x, 'grid_size_y', grid_size_y)

    grid_size_y += 1
    grid_size_x += 1

    x_world = np.arange(-(grid_size_x - 1) / 2, (grid_size_x + 1) / 2).reshape(1, -1) * np.ones((grid_size_y, 1)) * scale
    y_world = np.arange(-(grid_size_y - 1) / 2, (grid_size_y + 1) / 2).reshape(-1, 1) * np.ones((1, grid_size_x)) * scale

    mask = np.ones((grid_size_y, grid_size_x))
    mask[1:-1, 1:-1] = 0
    mask = (mask >= 0.5)

    x_world = x_world[mask]
    y_world = y_world[mask] + Y

    z_world = np.zeros(((grid_size_x + grid_size_y) * 2 - 4)) + Z


    plane_points = np.concatenate((x_world.reshape(-1, 1), y_world.reshape(-1, 1), z_world.reshape(-1, 1)), axis=1)
    return plane_points


def plot_edge_points(plane_points, ax, color=None ,grid_size_x=None, grid_size_y=None):
    '''
    INPUTS
    world_points - matrix of world points, each line is (x1, y1, z1)

    OUTPUTS - None
    '''
    if grid_size_x is None:
        grid_size_x = 10
    if grid_size_y is None:
        grid_size_y = 10
    if color is None:
        color = 'r'

    grid_size_y += 1
    grid_size_x += 1

    no_points = np.shape(plane_points)[0]

    alpha = 0.5

    # horizontal lines
    for j in range(grid_size_x):
        ax.plot((plane_points[j, 0], plane_points[j + no_points - grid_size_x, 0]),
                (plane_points[j, 1], plane_points[j + no_points - grid_size_x, 1]), color ,alpha=alpha)

    # vertical edges
    for j in [0, no_points - grid_size_x]:
        ax.plot((plane_points[j, 0], plane_points[j + grid_size_x - 1, 0]),
                (plane_points[j, 1], plane_points[j + grid_size_x - 1, 1]), color , alpha=alpha)

    # vertical middle
    for j in range(0, (grid_size_y - 2) * 2, 2):
        ax.plot((plane_points[j + grid_size_x, 0], plane_points[j + grid_size_x + 1, 0]),
                (plane_points[j + grid_size_x, 1], plane_points[j + grid_size_x + 1, 1]), color , alpha=alpha)


def plot_edge_points_3d(plane_points, ax, color=None, grid_size_x=None, grid_size_y=None):
    '''
    INPUTS
    world_points - matrix of world points, each line is (x1, y1, z1)

    OUTPUTS - None
    '''
    if grid_size_x is None:
        grid_size_x = 10
    if grid_size_y is None:
        grid_size_y = 10
    if color is None:
        color = 'r'

    grid_size_y += 1
    grid_size_x += 1

    no_points = np.shape(plane_points)[0]

    # horizontal lines
    for j in range(grid_size_x):
        ax.plot((plane_points[j, 0], plane_points[j + no_points - grid_size_x, 0]),
                (plane_points[j, 1], plane_points[j + no_points - grid_size_x, 1]),
                (plane_points[j, 2], plane_points[j + no_points - grid_size_x, 2]), color)

    # vertical edges
    for j in [0, no_points - grid_size_x]:
        ax.plot((plane_points[j, 0], plane_points[j + grid_size_x - 1, 0]),
                (plane_points[j, 1], plane_points[j + grid_size_x - 1, 1]),
                (plane_points[j, 2], plane_points[j + grid_size_x - 1, 2]), color)

    # vertical middle
    for j in range(0, (grid_size_y - 2) * 2, 2):
        ax.plot((plane_points[j + grid_size_x, 0], plane_points[j + grid_size_x + 1, 0]),
                (plane_points[j + grid_size_x, 1], plane_points[j + grid_size_x + 1, 1]),
                (plane_points[j + grid_size_x, 2], plane_points[j + grid_size_x + 1, 2]), color)


def transform_plane_points(plane_points, v, c):
    '''
    INPUTS
    plane_points - world points to be transformed, each line is (x1, y1, z1)
    V - vector of the unit normal of the plane ()
    C - the distance between the plane and the origin

    OUTPUTS
    transformed points - points of plane after tranfroamtion
    '''

    plane_norm = np.array((0, 0, 1))

    V = np.cross(v, plane_norm)

    # print('V[2]' , V[2])

    Vx = np.array([[0, -V[2], V[1]],
                   [V[2], 0, -V[0]],
                   [-V[1], V[0], 0]])

    R = np.eye(3) + Vx + np.matmul(Vx, Vx) / (1 + np.dot(v, plane_norm)) # find rotation matrix to rotate plane points to found plane

    rotated_points = np.matmul(plane_points, R) # Rotate plane to be parallel to found plane

    transformed_points = rotated_points + (c * v).T # Transfrom plane to lye on found groun plane

    Zvec = np.cross(v, np.array((1,0,0))) # find direction along plane with largest z component
    if Zvec[2] < 0:
        Zvec = -Zvec
    '''
    theta = 2 * np.pi / 180 * np.pi # correct for systemic error

    transformed_points = np.matmul(transformed_points, np.array(([1, 0,             0               ],
                                                              [0, np.cos(theta), -np.sin(theta)  ],
                                                              [0, np.sin(theta), np.cos(theta)   ])))
    '''

    for i in range(10000): # shift plan point to lie in front of the camera
        transformed_points += Zvec * 0.02
        if (transformed_points[:,2] > 0).all():
            transformed_points += Zvec * 0.02
            # print('I', i)
            break

    return transformed_points


def remove_systematic_error(transformed_points, deg):
    '''
    Correct for systemic error in plane orientation

    INPUTS
    transformed points - points of plane
    deg - correction amount in degrees

    OUTPUTS
    transformed points - points of plane after tranfroamtion
    '''
    theta = deg / 180 * np.pi # correct for systemic error

    transformed_points = np.matmul(transformed_points, np.array(([1, 0,             0               ],
                                                                 [0, np.cos(theta), -np.sin(theta)  ],
                                                                 [0, np.sin(theta), np.cos(theta)   ])))
    return transformed_points


def split_points(point_matrix):
    x = point_matrix[:, 0]
    y = point_matrix[:, 1]
    z = point_matrix[:, 2]
    return x, y, z


def plane_points_to_im(plane_world_points, down_sample=None, K=None):
    '''
    INPUTS
    plane_world_points - matrix of world points, each line is (x1, y1, z1)
    K - Intersic matrix of the camera

    OUTPUTS
    plane_image_points - matrix of world points, each line is (x1, y1, z1)

    '''
    if K is None:
        h_ds = 1920 / (512/down_sample)  # horizontal downs ample 3.75
        v_ds = 1080 / (256/down_sample)  # vertical downs ample 4.21875

        print('h', h_ds,'v' , v_ds)

        K = np.array([[1.40597825e+03 / h_ds, 0.00000000e+00, 9.69946177e+02 / h_ds],
                      [0.00000000e+00, 1.40675654e+03 / v_ds, 5.58178743e+02 / v_ds],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    if down_sample is None:
        down_sample = 1

    u_w, v_w, w_w = split_points(plane_world_points)

    x_im = (K[0, 0] * u_w / (w_w + 1e-9)) + K[0, 2]
    y_im = (K[1, 1] * v_w / (w_w + 1e-9)) + K[1, 2]

    x_im = x_im.reshape(-1, 1)
    y_im = y_im.reshape(-1, 1)

    plane_image_points = np.concatenate((x_im, y_im), axis=1)

    return plane_image_points


def down_sample(z_world, y_world, x_world, mask, ds_factor):

    if ds_factor is None:
        ds_factor = 1

    disp_width = z_world.shape[1]
    disp_height = z_world.shape[0]

    mask_width = mask.shape[1]
    mask_height = mask.shape[0]

    assert mask_width == disp_width
    assert mask_height == disp_height

    plot_w = disp_width // ds_factor
    plot_h = disp_height // ds_factor

    disparities_ds = np.concatenate((cv2.resize(x_world, (plot_w, plot_h)).reshape(-1, 1),
                                  cv2.resize(y_world, (plot_w, plot_h)).reshape(-1, 1),
                                  cv2.resize(z_world, (plot_w, plot_h)).reshape(-1, 1)), axis=1)


    if mask is None:
        mask = np.hstack((np.zeros((disp_height // 2, disp_width)), np.ones((disp_height // 2, disp_width))))

    mask_ds = cv2.resize(mask, (plot_w, plot_h))
    mask_ds = (mask_ds.astype(int) >= 0.5)

    mask_ds = mask_ds.reshape(-1, 1)

    return disparities_ds, mask_ds, plot_w, plot_h


def plot_ground_on_image(disparities, path_in, path_out, mask=None):
    '''
    INPUTS
    disparities - (w, h) matrix disparities (w, h)
    mask - 0, 1 mask of ground pixels
    image - original image
    path_out - path where to save image

    OUTPUTS - None
    '''

    # Parameters

    down_sample_amount = 4
    grid_x = 200
    grid_y = 60
    iters = 2500
    RANSAC_thres = 0.2
    grid_scale = 2
    mask_thres = 10
    ground_thres = 9

    world_points, z_world, y_world, x_world = dispartiy_to_world_points(disparities)

    # depth = 1 / (disparities + 1e-6)
    # depth = normalize_depth_for_display(depth, pc=99, crop_percent=0, normalizer=None, cmap='plasma')

    world_points_ds, mask_ds, plot_w, plot_h = down_sample(z_world, y_world, x_world, mask, ds_factor=down_sample_amount)
    down_sample_pixels = plot_w * plot_h


    mask_plot = mask_ds

    near_points = world_points_ds[:,2] <= 40
    near_points = near_points.reshape(-1, 1)

    mask = (mask_ds * near_points) >= 0.5

    no_mask_points = np.sum(np.sum(mask_ds.astype(int)))
    mask_precentage = no_mask_points * 100 / down_sample_pixels
    print('Number of masked pixels:', no_mask_points,
          'Total number of pixles:', plot_w * plot_h,
          'Percentage: ', mask_precentage)


    v, c, ground_mask = ransac_plane_with_mask(world_points_ds, mask_ds, iters, RANSAC_thres)

    # print(np.shape(v)) 
    # v = remove_systematic_error(v, 2)


    no_ground_points = np.sum(np.sum(ground_mask.astype(int)))
    ground_precentage =  no_ground_points * 100 / down_sample_pixels
    print('Number of inliers:', no_ground_points,
          'total number of pixles:', plot_w * plot_h,
          'Percentage: ', ground_precentage)

    occ_mask_2 = occlusion_mask(world_points, v, c, RANSAC_thres)
    # occ_mask_2 = ((occ_mask + ground_mask) >= 0.5).astype(int)

    plane_points = gen_edge_points(Z=0, Y=0, scale=2, grid_size_x=grid_x, grid_size_y=grid_y)

    horizon_points = np.array(([-1e6,1e5,0],[1e6,1e5,0]))

    transformed_points = transform_plane_points(plane_points, v, c)

    horizon_points = transform_plane_points(horizon_points, v, c)

    # transformed_points = remove_systematic_error(transformed_points, correct_deg)

    im_point = plane_points_to_im(transformed_points, down_sample=1)

    horizon_im = plane_points_to_im(horizon_points, down_sample=1)

    # ==================================================================================================================


    # Plot Masks
    '''    

    fig = plt.figure()
    ax_1 = fig.add_subplot(221)
    ax_2 = fig.add_subplot(222)
    ax_3 = fig.add_subplot(223)
    ax_4 = fig.add_subplot(224)
    # ax_5 = fig.add_subplot(235)
    #a x_6 = fig.add_subplot(236)
    ax_1.imshow(-np.clip(world_points_ds[:,2], -10, 10).reshape(plot_h,plot_w))
    # ax_2.imshow(mask_plot.reshape(plot_h,plot_w))
    ax_2.imshow(mask_ds.reshape(plot_h,plot_w))
    ax_3.imshow(ground_mask.reshape(plot_h, plot_w))
    ax_4.imshow(occ_mask_2.reshape(256, 512))
    # ax_6.imshow(occ_mask_2.reshape(plot_h, plot_w))
    ax_1.axis('off')
    ax_2.axis('off')
    ax_3.axis('off')
    ax_4.axis('off')
    # ax_5.axis('off')
    # ax_6.axis('off')
    ax_1.set_title('Depth')
    ax_2.set_title('Input Mask')
    ax_3.set_title('Output Mask')
    ax_4.set_title('Occlusion Mask')
    # ax_5.set_title('Occlusion Mask')
    # ax_6.set_title('Occ Mask 2')

    fig.savefig(path_out[:-4]+'masks', bbox_inches='tight')
    '''
    # ==================================================================================================================

    # Plot Scatters
    '''
    # world_points = np.clip(world_points, -40, 40)
    #
    # ground_points = (world_points[np.repeat(ground_mask, 3, axis=1)]).reshape(-1,3)
    #
    # not_ground_mask = (1 - ground_mask.astype(int)) >= 0.5
    #
    # not_ground_points = (world_points[np.repeat(not_ground_mask, 3, axis=1)]).reshape(-1,3)
    #
    #
    # fig_2 = plt.figure()
    #
    # alpha = 0.1
    # ax__1 = fig_2.add_subplot(131, projection='3d')
    # ax__2 = fig_2.add_subplot(132, projection='3d')
    # ax__3 = fig_2.add_subplot(133, projection='3d')
    # ax__1.scatter(not_ground_points[:, 0], not_ground_points[:, 1], not_ground_points[:, 2], color='g', marker='.', alpha=alpha)
    # ax__1.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], color='b', marker='.', alpha=alpha)
    # ax__1.view_init(azim=90, elev=90)
    # ax__1.set_ylabel('y')
    # ax__1.set_xlabel('x')
    #
    # ax__2.scatter(not_ground_points[:, 0], not_ground_points[:, 1], not_ground_points[:, 2], color='g', marker='.', alpha=alpha)
    # ax__2.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], color='b', marker='.', alpha=alpha)
    # ax__2.view_init(azim=00, elev=00)
    # ax__2.set_ylabel('y')
    # ax__2.set_xlabel('x')
    #
    # ax__3.scatter(not_ground_points[:, 0], not_ground_points[:, 1], not_ground_points[:, 2], color='g', marker='.', alpha=alpha)
    # ax__3.scatter(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], color='b', marker='.', alpha=alpha)
    # ax__3.view_init(azim=90, elev=00)
    # ax__3.set_ylabel('y')
    # ax__3.set_xlabel('x')
    #
    # plot_edge_points_3d(transformed_points, ax__1, 'r', grid_size_x=grid_x, grid_size_y=grid_y)
    # plot_edge_points_3d(transformed_points, ax__2, 'r', grid_size_x=grid_x, grid_size_y=grid_y)
    # plot_edge_points_3d(transformed_points, ax__3, 'r', grid_size_x=grid_x, grid_size_y=grid_y)
    #
    # fig_2.savefig(path_out[:-4]+'scatter', bbox_inches='tight')
    '''
    # ==================================================================================================================


    # get the size in inches
    dpi = 100
    # xinch = plot_w / dpi # plot_w = 512
    # yinch = plot_h / dpi # plot_h = 256


    xinch = 512 / dpi # plot_w = 512
    yinch = 256 / dpi # plot_h = 256
    # f = plt.figure()
    # f = plt.figure(frameon=False, figsize=(xinch, yinch))
    # ax = f.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    # ax.imshow(im, extent=[0, 512, 0, 256], interpolation='none')
    # ax.set_ylim([0,256])
    # ax.set_xlim([0,512])
    # ax.axis('off')
    # plot_edge_points(im_point, ax, grid_size_x=grid_x, grid_size_y=grid_y)
    # print('path_out', path_out)
    # f.canvas.print_png(path_out)

    # convent plot line to rgb image

    fig = plt.figure(frameon=False, figsize=(5.12, 2.56))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    plot_edge_points(im_point, ax, grid_size_x=grid_x, grid_size_y=grid_y)
    ax.set_ylim([0,256])
    ax.set_xlim([0,512])
    # ax.axis('off')

    fig.canvas.print_png('lines.png')
    lines = cv2.imread('lines.png', cv2.IMREAD_UNCHANGED)

    # blur mask and select pixels that lye below ground plane

    mask = occ_mask_2.astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (5, 5), 1)

    lines[:, :, 3] = lines[:, :, 3] * mask.reshape(256, 512)
    lines[:, :, 0] = lines[:, :, 2]
    lines[:, :, 2] = 0
    lines = np.flipud(lines)

    # print original image with ground plane superimposed

    im = np.array(cv2.imread(path_in))

    fig = plt.figure(frameon=False, figsize=(5.12, 2.56))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.axis('off')
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), extent=[0, 512, 0, 256], interpolation='none')

    if mask_precentage >= mask_thres and ground_precentage >= ground_thres :
        ax.imshow(lines)
        ax.plot((horizon_im[0, 0], horizon_im[1, 0]), (horizon_im[0, 1], horizon_im[1, 1]), 'b', alpha=1)
        v = np.round(v, 3)
        c = np.abs(np.round(c, 3)).reshape(-1)

        string = 'v=' + str(v) + ' d=' + str(c)
        ax.text(0.95, 0.01, string, verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color='blue', fontsize=10)

    if mask_precentage >= mask_thres and ground_precentage >= ground_thres :
        ax.imshow(lines)
        ax.plot((horizon_im[0, 0], horizon_im[1, 0]), (horizon_im[0, 1], horizon_im[1, 1]), 'b', alpha=1)
    else:
        if mask_precentage < mask_thres:
            ax.text(0.95, 0.01, 'No plane found - insufficient ground pixels',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color='red', fontsize=15)
        else:
            ax.text(0.95, 0.01, 'No plane found - no planar surface detected',
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,

                color='red', fontsize=15)



    ax.set_ylim([0, 256])
    ax.set_xlim([0, 512])
    # fig.canvas.print_png(path_out[:-4]+'.png')
    fig.canvas.print_png(path_out[:-4]+'.png')

    # plt.show()


def plot_ground(im_folder_path, mask_path, depth_path, out_folder):
    im_names = []
    for filename in os.listdir(im_folder_path):
        im_names.append(filename)
    im_names.sort()

    print('Number of images: ', len(im_names))

    depths = np.load(depth_path),
    print('Shape Depths: ', np.shape(depths))

    masks = np.load(mask_path)
    print('Shape Masks: ', np.shape(masks))

    for i in range(len(im_names)):
        im_path = os.path.join(im_folder_path, im_names[i])

        path_out = os.path.join(out_folder, im_names[i])

        disparities = depths[0][i, :, :]

        mask = masks[:, :, i]

        t = time.time()
        print('Processing: ', im_names[i])
        plot_ground_on_image(disparities, im_path, path_out, mask)
        print('time:' , time.time() - t )


if __name__ == "__main__":

    # this should run ground plane estimation system on example image

    main_ft.main()

    plot_ground('/Data/test_ims/example_img.jpg',
      '/Data/inter/masks.npy',
      '/Data/inter/disparities.npy',
      '/Data/outputs/')



