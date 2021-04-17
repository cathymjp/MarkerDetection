import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import imutils
from scipy.spatial import distance as dist
from imutils import contours
from imutils import perspective
from math import atan2,degrees
import statistics


# point ordering list
list_one = [0, 1, 2, 3]
list_two = [2, 3, 1, 0]


def resize_image(image):
    scale_percent = 17  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return image


def apply_threshold(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([30, 58, 140], np.uint8)
    yellow_upper = np.array([250, 255, 255], np.uint8)

    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    pink_lower = np.array([162, 30, 190], np.uint8)
    pink_upper = np.array([255, 150, 255], np.uint8)

    pink = cv2.inRange(hsv, pink_lower, pink_upper)

    image_res1 = cv2.bitwise_and(image, image, mask=yellow)
    image_res = cv2.cvtColor(image_res1, cv2.COLOR_BGR2RGB)

    image_res_thre = cv2.cvtColor(image_res, cv2.COLOR_RGB2GRAY)
    _, image_res_thre = cv2.threshold(image_res_thre, 255, 255, cv2.THRESH_OTSU)
    # cv2.imshow("image_res_thre_", image_res_thre)

    return image_res1, image_res_thre


def midpoint(x, y):
    return (x[0] + y[0]) * 0.5, (x[1] + y[1]) * 0.5


def calculate_distance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return dist


def distance_to_camera(known_width, focal_length, pic_width):
    return (known_width * focal_length) / pic_width


def angle_change(initial, moved):
    x_change = moved[0] - initial[0]
    y_change = moved[1] - initial[1]

    return degrees(atan2(y_change, x_change))


def order_points(pts):
    # initialize the list in top-left, top-right, bottom-right, and bottom-left order
    rect = np.zeros((4, 2), dtype="float32")

    # top-left and bottom-right point computation
    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]  # 0 original
    rect[2] = pts[np.argmax(s)]  # 2 original

    # top-right and bottom-left point computation
    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]  # 1 original
    rect[3] = pts[np.argmax(diff)]  # 3 original

    return rect


# Order points from smallest values in point order list
def compare_lists(list_to_compare):
    list_order = []

    for i in range(0, 4):
        list_order.append(list_to_compare[0][i][0] + list_to_compare[0][i][1])

    # give list values sorted index values
    sorted_index = np.argsort(np.argsort(list_order))

    return sorted_index


def grab_contour(threshold_image):
    # sort the contours from left-to-right and initialize the 'pixels per metric' calibration variable
    cnts = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the bounding box point colors
    (cnts, _) = contours.sort_contours(cnts)
    new_swap_list = []
    leftmost_contour = None
    center_points, areas, distances, corners, three_areas, coords, testing = [], [], [], [], [], [], []

    known_width = 7.6
    focal_length = 300

    for (i, c) in enumerate(cnts):
        area = cv2.contourArea(c)
        three_areas.append(area)
        sorteddata = sorted(zip(three_areas, cnts), key=lambda x: x[0], reverse=True)
        if cv2.contourArea(c) <= 30:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        cv2.drawContours(threshold_image, [box], -1, (0, 255, 0), 2)
        rect = order_points(box)

        testing.append(rect)
        coords.append(rect[0])

    index_ordered_list = []     # whole point values in ordered index values

    # Four largest contours' coordinates
    compare_list = [sorteddata[0][1][0][0], sorteddata[1][1][0][0], sorteddata[2][1][0][0], sorteddata[3][1][0][0]]
    first, second, third, fourth = compare_lists([compare_list])

    index_ordered_list.extend((first, second, third, fourth))

    for i in np.argsort(index_ordered_list):
        new_swap_list.append(sorteddata[i][1])

    for c in new_swap_list:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        area = cv2.contourArea(c)

        if cv2.contourArea(c) <= 30:
            continue

        box = approx
        box = np.squeeze(box)

        # order the points in the contour and draw outlines of the rotated rounding box
        box = order_points(box)
        box = perspective.order_points(box)
        testing.append(box)

        (x, y, w, h) = cv2.boundingRect(c)

        # compute area
        area = cv2.contourArea(c)
        areas.append(area)

        # compute center points
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        center = (cx, cy)
        center_points.append(center)

        c_x = np.average(box[:, 0])
        c_y = np.average(box[:, 1])

        # compute corners from contour image
        # four_corners = corners_from_contour(threshold_image, c)
        corners.append(box)

        # compute and return the distance from the maker to the camera
        distances.append(distance_to_camera(known_width, focal_length, w))

        if leftmost_contour is None:
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # compute the Euclidean distance between the midpoints, then construct the reference object
            d = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            leftmost_contour = (box, (c_x, c_y), d / 7.5)
            # first_box = box
            continue

    # swap to order center_points
    if center_points[1][0] <= center_points[2][0]:
        tmp = center_points[2]
        center_points[2] = center_points[1]
        center_points[1] = tmp

    if corners[1][0][0] <= corners[2][0][0]:
        tmp = corners[2]
        corners[2] = corners[1]
        corners[1] = tmp

    return leftmost_contour, center_points, areas, distances, corners


# two image difference computations and results
def marker_calculation():
    reference_marker_image = cv2.imread('images/1114_15m_rotL_0.jpg')
    moved_marker_image = cv2.imread('images/1114_15m_rotL_2.jpg')

    cv2.imshow("Resized ref image", resize_image(reference_marker_image))
    cv2.imshow("Resized moved image", resize_image(moved_marker_image))

    res_ref1, reference_marker_image = apply_threshold(resize_image(reference_marker_image))
    res_moved1, moved_marker_image = apply_threshold(resize_image(moved_marker_image))
    cv2.imshow("res_Ref1", res_ref1)
    cv2.imshow("reference_marker_image", reference_marker_image)

    # Center of image
    center_X = int(reference_marker_image.shape[1] / 2)
    center_Y = int(reference_marker_image.shape[0] / 2)

    ref_leftmost_contour, ref_center_point, ref_areas, ref_distances2, ref_corners = grab_contour(
        reference_marker_image)
    moved_leftmost_contour, moved_center_point, moved_areas, moved_distances2, moved_corners = grab_contour(
        moved_marker_image)

    # place blue dots on four points used
    for one, two in zip(list_one, list_two):
        cv2.circle(res_ref1, (ref_corners[one][two][0], ref_corners[one][two][1]), 1, (255, 140, 65), thickness=5,
                   lineType=8, shift=0)
        cv2.circle(res_moved1, (moved_corners[one][two][0], moved_corners[one][two][1]), 1, (255, 140, 65), thickness=5,
                   lineType=8, shift=0)
    cv2.imshow("reference points color", res_ref1)
    cv2.imshow("moved points color", res_moved1)

    angle_difference, center_difference, corners_difference, distance_difference = [], [], [], []
    center_testing, angle_difference_reverse = [], []

    # Difference calculations
    print("====== Differences ======")
    for i in range(0, len(ref_center_point)):
        # center angle change
        angle_difference.append(angle_change(ref_center_point[i], moved_center_point[i]))
        angle_difference_reverse.append(angle_change(moved_center_point[i], ref_center_point[i]))

        # center point change
        center_difference.append(math.sqrt((ref_center_point[i][0] - moved_center_point[i][0]) ** 2 + (
                ref_center_point[i][1] - moved_center_point[i][1]) ** 2))

        # markers' camera distance difference
        # distance_difference.append(ref_distances[i] - moved_distances[i])

        center_testing.append(ref_center_point[i][0] - moved_center_point[i][0])
        center_testing.append(ref_center_point[i][1] - moved_center_point[i][1])

    ref_x_changes, ref_y_changes, ref_tangents, moved_x_changes, moved_y_changes, moved_tangents = \
        [], [], [], [], [], []

    for one, two in zip(list_one, list_two):
        ref_x_changes.append(ref_corners[one][two][0] - center_X)
        ref_y_changes.append(ref_corners[one][two][1] - center_Y)
        moved_x_changes.append(moved_corners[one][two][0] - center_X)
        moved_y_changes.append(moved_corners[one][two][1] - center_Y)

    total_x = 0.0
    total_y = 0.0

    if ref_x_changes[0] * ref_x_changes[1] < 0 or ref_x_changes[0] * ref_x_changes[1] == 0:
        # marker's first corner tangent inverse (outer angle)
        for i in range(0, 4):
            ref_tangents.append(math.degrees(math.atan(ref_x_changes[i] / ref_y_changes[i])))
        for i in range(0, 4):
            ref_tangents.append(math.degrees(math.atan(ref_y_changes[i] / ref_x_changes[i])))

        # print("ref_tangents: ", ref_tangents)

        ref_summation_up = ref_tangents[0] + abs(ref_tangents[1])
        ref_summation_down = abs(ref_tangents[2]) + abs(ref_tangents[3])
        ref_summation_left = abs(ref_tangents[4]) + abs(ref_tangents[6])
        ref_summation_right = abs(ref_tangents[5]) + abs(ref_tangents[7])

    moved_tangents_test_X, moved_tangents_test_Y, moved_tangents_test, adjustments_top_test = [], [], [], []

    print("Dist1: Ref Y coordinate Top", (center_Y - (ref_corners[0][2][1] + ref_corners[1][3][1]) / 2))
    print("Dist2: Ref Y coordinate Bottom", ((ref_corners[2][1][1] + ref_corners[3][0][1]) / 2) - center_Y)
    print("Dist3: Ref X coordinate Left", (center_X - (ref_corners[0][2][0] + ref_corners[2][1][0]) / 2))
    print("Dist4: Ref X coordinate Right", ((ref_corners[1][3][0] + ref_corners[3][0][0]) / 2) - center_X)

    # tangent calculation for angle change
    for i in range(0, 4):
        moved_tangents_test_X.append(math.degrees(math.atan(moved_x_changes[i] / moved_y_changes[i])))
        moved_tangents_test.append(math.degrees(math.atan(moved_x_changes[i] / moved_y_changes[i])))
        moved_tangents_test_Y.append(math.degrees(math.atan(moved_y_changes[i] / moved_x_changes[i])))
    print("moved_tangents_test_X: ", moved_tangents_test_X)
    print("moved_tangents_test_Y: ", moved_tangents_test_Y)

    if moved_tangents_test_X[0] * moved_tangents_test_X[1] >= 0:
        summation_testing_up_test = abs(moved_tangents_test_X[0] - moved_tangents_test_X[1])
        summation_testing_down_test = abs(moved_tangents_test_X[2] - moved_tangents_test_X[3])
    else:
        summation_testing_up_test = abs(moved_tangents_test_X[1] - moved_tangents_test_X[0])
        summation_testing_down_test = abs(moved_tangents_test_X[2] - moved_tangents_test_X[3])

    summation_testing_left_test = abs(moved_tangents_test_Y[0] - moved_tangents_test_Y[2])
    summation_testing_right_test = abs(moved_tangents_test_Y[1] - moved_tangents_test_Y[3])

    x_ratio, y_ratio = [], []

    # angle ratios of the original image
    moved_top_left_ratio_test = (ref_tangents[0] / ref_summation_up) * summation_testing_up_test
    x_ratio.append((ref_tangents[0] / ref_summation_up) * summation_testing_up_test)
    moved_top_right_ratio_test = (ref_tangents[1] / ref_summation_up) * summation_testing_up_test
    x_ratio.append((ref_tangents[1] / ref_summation_up) * summation_testing_up_test)
    moved_bottom_left_ratio_test = (ref_tangents[2] / ref_summation_down) * summation_testing_down_test
    x_ratio.append((ref_tangents[2] / ref_summation_down) * summation_testing_down_test)
    moved_bottom_right_ratio_test = (ref_tangents[3] / ref_summation_down) * summation_testing_down_test
    x_ratio.append((ref_tangents[3] / ref_summation_down) * summation_testing_down_test)

    moved_left_top_ratio_test = (ref_tangents[4] / ref_summation_left) * summation_testing_left_test
    y_ratio.append((ref_tangents[4] / ref_summation_left) * summation_testing_left_test)
    moved_right_top_ratio_test = (ref_tangents[5] / ref_summation_right) * summation_testing_right_test
    y_ratio.append((ref_tangents[5] / ref_summation_right) * summation_testing_right_test)
    moved_left_bottom_ratio_test = (ref_tangents[6] / ref_summation_left) * summation_testing_left_test
    y_ratio.append((ref_tangents[6] / ref_summation_left) * summation_testing_left_test)
    moved_right_bottom_ratio_test = (ref_tangents[7] / ref_summation_right) * summation_testing_right_test
    y_ratio.append((ref_tangents[7] / ref_summation_right) * summation_testing_right_test)

    testing_y_index, testing_x_index = [], []
    print("x_ratio", x_ratio)
    print("y_ratio", y_ratio)

    for i in range(0, 4):
        if moved_tangents_test_X[i] * x_ratio[i] < 0:
            testing_x_index.insert(i, moved_tangents_test_X[i] + x_ratio[i])
        if moved_tangents_test_Y[i] * y_ratio[i] < 0:
            testing_y_index.insert(i, moved_tangents_test_Y[i] + y_ratio[i])
        else:
            testing_x_index.insert(i, x_ratio[i] - moved_tangents_test_X[i])
            testing_y_index.insert(i, y_ratio[i] - moved_tangents_test_Y[i])

    testing_x_index[1] = testing_x_index[1] * -1
    testing_x_index[2] = testing_x_index[2] * -1
    testing_y_index[1] = testing_y_index[1] * -1
    testing_y_index[2] = testing_y_index[2] * -1

    print("Gamma: testing_x_index_value", testing_x_index)
    for i in testing_x_index:
        total_x += abs(i)
    print("TOTAL: X", total_x / 4)

    print("Gamma: testing_y_index_value", testing_y_index)
    for i in testing_y_index:
        total_y += abs(i)
    print("TOTAL: Y", total_y / 4)

    # length from original points to center point
    orig_m = calculate_distance(ref_corners[0][2][0], ref_corners[0][2][1], center_X, center_Y)
    orig_n = calculate_distance(ref_corners[1][3][0], ref_corners[1][3][1], center_X, center_Y)

    orig_p = calculate_distance(ref_corners[2][1][0], ref_corners[2][1][1], center_X, center_Y)
    orig_q = calculate_distance(ref_corners[3][0][0], ref_corners[3][0][1], center_X, center_Y)

    orig_f = calculate_distance(ref_corners[0][2][0], ref_corners[0][2][1], center_X, center_Y)
    orig_g = calculate_distance(ref_corners[2][1][0], ref_corners[2][1][1], center_X, center_Y)

    orig_s = calculate_distance(ref_corners[1][3][0], ref_corners[1][3][1], center_X, center_Y)
    orig_t = calculate_distance(ref_corners[3][0][0], ref_corners[3][0][1], center_X, center_Y)

    # Original angles to randians
    orig_alpha = math.sin(abs(math.radians(ref_tangents[0])))
    orig_betha = math.sin(abs(math.radians(ref_tangents[1])))
    orig_alpha2 = math.sin(abs(math.radians(ref_tangents[2])))
    orig_betha2 = math.sin(abs(math.radians(ref_tangents[3])))

    # Original M points for reference image
    orig_new_x_coordinate = int(
        (orig_m * orig_alpha * ref_corners[1][3][0] + orig_n * orig_betha * ref_corners[0][2][0]) /
        (orig_m * orig_alpha + orig_n * orig_betha))
    orig_new_y_coordinate = int(
        (orig_m * orig_alpha * ref_corners[1][3][1] + orig_n * orig_betha * ref_corners[0][2][1]) / (
                orig_m * orig_alpha + orig_n * orig_betha))
    orig_new_x_coordinate1 = int(
        (orig_p * orig_alpha * ref_corners[3][0][0] + orig_q * orig_betha * ref_corners[2][1][0]) /
        (orig_p * orig_alpha + orig_q * orig_betha))
    orig_new_y_coordinate1 = int(
        (orig_p * orig_alpha * ref_corners[3][0][1] + orig_q * orig_betha * ref_corners[2][1][1]) /
        (orig_p * orig_alpha + orig_q * orig_betha))

    orig_new_x_coordinate2 = int(
        (orig_f * orig_alpha2 * ref_corners[2][1][0] + orig_g * orig_betha2 * ref_corners[0][2][0]) /
        (orig_f * orig_alpha2 + orig_g * orig_betha2))
    orig_new_y_coordinate2 = int(
        (orig_f * orig_alpha2 * ref_corners[2][1][1] + orig_g * orig_betha2 * ref_corners[0][2][1]) /
        (orig_f * orig_alpha2 + orig_g * orig_betha2))

    orig_new_x_coordinate3 = int(
        (orig_s * orig_alpha2 * ref_corners[3][0][0] + orig_t * orig_betha2 * ref_corners[1][3][0]) /
        (orig_s * orig_alpha2 + orig_t * orig_betha2))
    orig_new_y_coordinate3 = int(
        (orig_s * orig_alpha2 * ref_corners[3][0][1] + orig_t * orig_betha2 * ref_corners[1][3][1]) /
        (orig_s * orig_alpha2 + orig_t * orig_betha2))

    # Distance from original M points to center point
    orig_top_M_dist = calculate_distance(center_X, center_Y, orig_new_x_coordinate, orig_new_y_coordinate)
    orig_bottom_M_dist = calculate_distance(center_X, center_Y, orig_new_x_coordinate1, orig_new_y_coordinate1)
    orig_left_M_dist = calculate_distance(center_X, center_Y, orig_new_x_coordinate2, orig_new_y_coordinate2)
    orig_right_M_dist = calculate_distance(center_X, center_Y, orig_new_x_coordinate3, orig_new_y_coordinate3)

    print("orig_new_x_coordinate", orig_new_x_coordinate)
    print("orig_new_y_coordinate", orig_new_y_coordinate)
    print("Original M coordinate top: ({}, {})".format(orig_new_x_coordinate, orig_new_y_coordinate))
    print("Original M coordinate bottom: ({}, {})".format(orig_new_x_coordinate1, orig_new_y_coordinate1))
    print("Original M coordinate left: ({}, {})".format(orig_new_x_coordinate2, orig_new_y_coordinate2))
    print("Original M coordinate right: ({}, {})".format(orig_new_x_coordinate3, orig_new_y_coordinate3))
    print("\n")
    print("Original M top dist: ", orig_top_M_dist)
    print("Original M bottom dist: ", orig_bottom_M_dist)
    print("Original M left dist: ", orig_left_M_dist)
    print("Original M right dist: ", orig_right_M_dist)

    # =========================== MOVED IMAGE CALCULATION ===================================
    # point distance calculations
    m = calculate_distance(moved_corners[0][2][0], moved_corners[0][2][1], center_X, center_Y)
    n = calculate_distance(moved_corners[1][3][0], moved_corners[1][3][1], center_X, center_Y)

    p = calculate_distance(moved_corners[2][1][0], moved_corners[2][1][1], center_X, center_Y)
    q = calculate_distance(moved_corners[3][0][0], moved_corners[3][0][1], center_X, center_Y)

    f = calculate_distance(moved_corners[0][2][0], moved_corners[0][2][1], center_X, center_Y)
    g = calculate_distance(moved_corners[2][1][0], moved_corners[2][1][1], center_X, center_Y)

    s = calculate_distance(moved_corners[1][3][0], moved_corners[1][3][1], center_X, center_Y)
    t = calculate_distance(moved_corners[3][0][0], moved_corners[3][0][1], center_X, center_Y)

    # Moved image angles to radians
    alpha = math.sin(abs(math.radians(moved_top_left_ratio_test)))
    betha = math.sin(abs(math.radians(moved_top_right_ratio_test)))
    alpha2 = math.sin(abs(math.radians(moved_right_top_ratio_test)))
    betha2 = math.sin(abs(math.radians(moved_right_bottom_ratio_test)))

    # Moved M points for moved image
    new_x_coordinate = int(
        (m * alpha * moved_corners[1][3][0] + n * betha * moved_corners[0][2][0]) / (m * alpha + n * betha))
    new_y_coordinate = int(
        (m * alpha * moved_corners[1][3][1] + n * betha * moved_corners[0][2][1]) / (m * alpha + n * betha))

    new_x_coordinate1 = int(
        (p * alpha * moved_corners[3][0][0] + q * betha * moved_corners[2][1][0]) / (p * alpha + q * betha))
    new_y_coordinate1 = int(
        (p * alpha * moved_corners[3][0][1] + q * betha * moved_corners[2][1][1]) / (p * alpha + q * betha))

    new_x_coordinate2 = int(
        (f * alpha2 * moved_corners[2][1][0] + g * betha2 * moved_corners[0][2][0]) / (f * alpha2 + g * betha2))
    new_y_coordinate2 = int(
        (f * alpha2 * moved_corners[2][1][1] + g * betha2 * moved_corners[0][2][1]) / (f * alpha2 + g * betha2))

    new_x_coordinate3 = int(
        (s * alpha2 * moved_corners[3][0][0] + t * betha2 * moved_corners[1][3][0]) / (s * alpha2 + t * betha2))
    new_y_coordinate3 = int(
        (s * alpha2 * moved_corners[3][0][1] + t * betha2 * moved_corners[1][3][1]) / (s * alpha2 + t * betha2))

    # M' change
    print("Moved M coordinate top: ({}, {})".format(new_x_coordinate, new_y_coordinate))
    print("Moved M coordinate bottom: ({}, {})".format(new_x_coordinate1, new_y_coordinate1))
    print("Moved M coordinate left: ({}, {})".format(new_x_coordinate2, new_y_coordinate2))
    print("Moved M coordinate right: ({}, {})".format(new_x_coordinate3, new_y_coordinate3))
    print("\n")

    moved_top_M_dist = calculate_distance(center_X, center_Y, new_x_coordinate, new_y_coordinate)
    moved_bottom_M_dist = calculate_distance(center_X, center_Y, new_x_coordinate1, new_y_coordinate1)
    moved_left_M_dist = calculate_distance(center_X, center_Y, new_x_coordinate2, new_y_coordinate2)
    moved_right_M_dist = calculate_distance(center_X, center_Y, new_x_coordinate3, new_y_coordinate3)

    print("Moved M top dist:", moved_top_M_dist)
    print("Moved M bottom dist:", moved_bottom_M_dist)
    print("Moved M left dist:", moved_left_M_dist)
    print("Moved M right dist:", moved_right_M_dist)

    distances = []
    total = 0.0

    print("M dist diff top:", orig_top_M_dist - moved_top_M_dist)
    print("M dist diff bottom:", orig_bottom_M_dist - moved_bottom_M_dist)
    print("M dist diff left:", orig_left_M_dist - moved_left_M_dist)
    print("M dist diff right:", orig_right_M_dist - moved_right_M_dist)

    distances.append(orig_top_M_dist - moved_top_M_dist)
    distances.append(orig_bottom_M_dist - moved_bottom_M_dist)
    distances.append(orig_left_M_dist - moved_left_M_dist)
    distances.append(orig_right_M_dist - moved_right_M_dist)

    for i in distances:
        total += i
    mean = total / len(distances)
    print("TOTAL:", total)
    print("Total distance moved: ", mean)
    print("\n")

    # X and Y re-location
    print("Coord diff top: ({}, {})".format(orig_new_x_coordinate - new_x_coordinate,
                                            orig_new_y_coordinate - new_y_coordinate))
    print("Coord diff bottom: ({}, {})".format(orig_new_x_coordinate1 - new_x_coordinate1,
                                               orig_new_y_coordinate1 - new_y_coordinate1))
    print("Coord diff left: ({}, {})".format(orig_new_x_coordinate2 - new_x_coordinate2,
                                             orig_new_y_coordinate2 - new_y_coordinate2))
    print("Coord diff right: ({}, {})".format(orig_new_x_coordinate3 - new_x_coordinate3,
                                              orig_new_y_coordinate3 - new_y_coordinate3))
    print("Average top/bottom diff:",
          ((orig_new_x_coordinate - new_x_coordinate) + (orig_new_x_coordinate1 - new_x_coordinate1)) / 2)
    print("Average left/right diff:",
          ((orig_new_y_coordinate2 - new_y_coordinate2) + (orig_new_y_coordinate3 - new_y_coordinate3)) / 2)


if __name__ == '__main__':
    marker_calculation()
    cv2.waitKey()
    cv2.destroyAllWindows
