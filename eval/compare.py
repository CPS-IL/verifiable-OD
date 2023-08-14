
import argparse
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import multiprocessing
import numpy as np
import os
import pprint
import signal
import sys
import tqdm

from scipy.spatial.transform import Rotation
from skspatial.objects import Line
from skspatial.objects import Plane
from skspatial.objects import Point
from skspatial.objects import Vector

import constants


# Globals, rather than passing arguments via multiprocessing
args = None
tqdm_position = multiprocessing.Value('i', 0)
OUTPUT_DIR = "top_views"
GT_ID_INDEX = 7

def get_argparser():
    parser = argparse.ArgumentParser()

    # Arguments worth looking at for each use
    parser.add_argument('--dataset', required=True,
                        help='Path to directory containing tfrecord files and extracted GT, Dets')
    parser.add_argument('--det-name', default='depth_clustering_detection_cube.json',
                        help='filenames for detection')
    parser.add_argument('--gt-name', default='waymo_ground_truth_cube_safety.json',
                        help='filename for ground truth')
    parser.add_argument('--outfile', type=str, default="top_view_coverage.json",
                        help='Name of the output file')
    parser.add_argument('--skip_existing', action="store_true",
                        help='Skip if output exists, off by default')
    parser.add_argument('--threads', '-j', type=int, default=os.cpu_count()//2,
                        help='Number of multiprocessing threads to use.')

    # Should generally be left to default
    parser.add_argument('--bb_long_edge_threshold', type=float, default=10,
                        help="Any detection BB with edge larger than this will be divided into two, recursive.")
    parser.add_argument('--dist_threshold_mult', type=float, default=1.05,
                        help="Multiplier for threshold distance for detection distance")
    parser.add_argument('--dist_threshold_add', type=float, default=0.1,
                        help="Addition to threshold distance for detection distance")

    # Debug friendly options
    parser.add_argument('--debug', action="store_true",
                        help="debug prints and single threaded")
    parser.add_argument('--segment', type=str, default=None,
                        help='Run only one segment')
    parser.add_argument('--frame', type=int, default=None,
                        help='Run only this frame')
    parser.add_argument('--gt_index', type=int, default=None,
                        help='Run only this object')
    parser.add_argument('--gt_id', type=str, default=None,
                        help='Run only this object')
    parser.add_argument('--save_images', action="store_true",
                        help='Whether to store top view images')
    return parser

def get_args():
    parser = get_argparser()
    args = parser.parse_args()
    args.dataset = os.path.realpath(args.dataset)

    if args.debug or args.save_images:
        args.threads = 1 # Debug prints would not make sense otherwise and matplotlib fails
        print ("[Info] Debug mode and Save Image forces single threaded execution")

    return args

def handle_sigint(signum, frame):
    print('%d %s handling signal %r' % (signum))
    sys.exit(-1)

# minDistance source https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
# Min distance from a line AB to a point E
def minDistance(A, B, E):

    # vector AB
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]

    # vector BP
    BE = [None, None]
    BE[0] = E[0] - B[0]
    BE[1] = E[1] - B[1]

    # vector AP
    AE = [None, None]
    AE[0] = E[0] - A[0]
    AE[1]   = E[1] - A[1]

    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]

    # Minimum distance from
    # point E to the line segment
    reqAns = 0

    # Case 1
    if (AB_BE > 0):
        y = E[1] - B[1]
        x = E[0] - B[0]
        reqAns = math.sqrt(x * x + y * y)
        reqP = B

    elif (AB_AE < 0):
        y = E[1] - A[1]
        x = E[0] - A[0]
        reqAns = math.sqrt(x * x + y * y)
        reqP = A

    else:
        l = Line.from_points(A, B)
        reqP = l.project_point(E)
        reqAns = reqP.distance_point(E)

    # [Distance, Point]
    return [reqAns, reqP]

# Find closest distance between two rectangles
def rectangle_find_closest(rec, rec_ego = constants.rec_ego):

    ego_min = []
    rec_min = []
    for i,_ in enumerate(rec_ego):
        for j,_ in enumerate(rec):
            if j == len(rec) - 1: j_next = 0
            else: j_next = j + 1
            if rec[j] == rec[j_next]: # Detection is not a rectangle, its a line or point.
                rec_min.append([Point(rec[j]).distance_point(rec_ego[i]), Point(rec[j])])
            else:
                rec_min.append(minDistance(rec[j], rec[j_next], rec_ego[i]))
    for i,_ in enumerate(rec):
        for j,_ in enumerate(rec_ego):
            if j == len(rec) - 1: j_next = 0
            else: j_next = j + 1
            ego_min.append(minDistance(rec_ego[j], rec_ego[j_next], rec[i]))

    min_rec = [9999, 0]
    min_ego = [9999, 0]

    for e in ego_min:
        if e[0] < min_ego[0]:
            min_ego = e
    for e in rec_min:
        if e[0] < min_rec[0]:
            min_rec = e

    if min_rec[0] <= min_ego[0]:
        # We already have the point on obstacle
        return min_rec[1]
    else:
        rec_min = []
        for j,_ in enumerate(rec):
            if j == len(rec) - 1: j_next = 0
            else: j_next = j + 1
            if rec[j] == rec[j_next]: # Detection is not a rectangle, its a line or point.
                rec_min.append([Point(rec[j]).distance_point(rec_ego[i]), Point(rec[j])])
            else:
                rec_min.append(minDistance(rec[j], rec[j_next], min_ego[1]))
        for e in rec_min:
            if e[0] < min_rec[0]:
                min_rec = e
        return min_rec[1]

def get_projected_segment(line, rec):
    pr = []
    for r in rec:
        pr.append(line.intersect_line(Line.from_points(r, [0, 0])))
    i_min = 0
    i_max = 0
    for i,_ in enumerate(pr):
        if pr[i][0] < pr[i_min][0]:
            i_min = i
        if pr[i][0] > pr[i_max][0]:
            i_max = i

    if i_min != i_max:
        return pr[i_min], pr[i_max]

    for i,_ in enumerate(pr):
        if pr[i][1] < pr[i_min][1]:
            i_min = i
        if pr[i][1] > pr[i_max][1]:
            i_max = i

    return pr[i_min], pr[i_max]


# SE3_from_R_t heading_to_rotmat apply_SE3 get_3d_bbox_egovehicle_frame
# Source: https://github.com/waymo-research/waymo-open-dataset/issues/192
# Convert Waymo's center, extent, heading into x,y,z format
def SE3_from_R_t(rotation, translation):
	dst_SE3_src = np.eye(4)
	dst_SE3_src[:3,:3] = rotation
	dst_SE3_src[:3,3] = translation
	return dst_SE3_src


def heading_to_rotmat(yaw):
	return Rotation.from_euler('z', yaw).as_matrix()


def apply_SE3(dst_SE3_src, pts_src):
	num_pts = pts_src.shape[0]
	homogeneous_pts = np.hstack([pts_src, np.ones((num_pts, 1))])
	dst_pts_h = homogeneous_pts.dot(dst_SE3_src.T)
	return dst_pts_h[:, :3] # remove homogeneous


def get_3d_bbox_egovehicle_frame(box):
    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = box[3] / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = box[4] / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = box[5] / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners_obj_frame = np.vstack((x_corners, y_corners, z_corners)).T

    ego_T_obj = np.array([ box[0], box[1], box[2] ])
    egovehicle_SE3_object = SE3_from_R_t(
        rotation=heading_to_rotmat(box[6]),
        translation=ego_T_obj,
    )
    egovehicle_pts = apply_SE3(egovehicle_SE3_object, corners_obj_frame)
    return egovehicle_pts


def top_from_box(pts):
    return [
        [pts[0][0], pts[0][1]],
        [pts[1][0], pts[1][1]],
        [pts[5][0], pts[5][1]],
        [pts[4][0], pts[4][1]],
    ]


def dets_break_too_large(args, det, ax):
    det = [det]
    while(1):
        break_det = False
        for i,d in enumerate(det):
            if max (Point(d[0]).distance_point(d[1]),
                    Point(d[1]).distance_point(d[2]),
                    Point(d[2]).distance_point(d[3]),
                    Point(d[3]).distance_point(d[0])) > args.bb_long_edge_threshold:
                break_det = True
                break
        if break_det == False: return det

        long = det.pop(i)
        # for j,l in enumerate(long):
            # ax.text(l[0], l[1], str(j))

        x_diff = abs(long[0][0] - long[2][0])
        y_diff = abs(long[0][1] - long[1][1])

        if x_diff > y_diff:
            det.append([long[0],
                        long[1],
                        [(long[1][0] + long[2][0])/2, (long[1][1] + long[2][1])/2],
                        [(long[3][0] + long[0][0])/2, (long[3][1] + long[0][1])/2]])
            det.append([[(long[3][0] + long[0][0])/2, (long[3][1] + long[0][1])/2],
                        [(long[1][0] + long[2][0])/2, (long[1][1] + long[2][1])/2],
                        long[2],
                        long[3]])
        else:
            det.append([long[0],
                        [(long[0][0] + long[1][0])/2, (long[0][1] + long[1][1])/2],
                        [(long[2][0] + long[3][0])/2, (long[2][1] + long[3][1])/2],
                        long[3]])
            det.append([[(long[0][0] + long[1][0])/2, (long[0][1] + long[1][1])/2],
                        long[1],
                        long[2],
                        [(long[2][0] + long[3][0])/2, (long[2][1] + long[3][1])/2]])


def get_tops(args, dets, ax):
    tops = []
    for det in dets[:-1]:
        det = [float(l) for l in det[:7]]
        pts = get_3d_bbox_egovehicle_frame(det)
        top = top_from_box(pts)
        top = dets_break_too_large(args, top, ax)
        tops = tops + top
    return tops


def plot_ego(ax):
    ax.add_patch(patches.Rectangle(
                 xy=(-1 * constants.ego_length / 2, -1 * constants.ego_width / 2),
                 width = constants.ego_length,
                 height = constants.ego_width,
                 linewidth = 1,
                 edgecolor = 'black',
                 facecolor = 'none'))
    ax.arrow(-.5, 0, 1, 0, width=.1)


def plot_one_box(ax, pts, color, width=1):
    if len(pts) == 8: # 3D points
        ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color=color, linestyle='-', linewidth=width)
        ax.plot([pts[1][0], pts[5][0]], [pts[1][1], pts[5][1]], color=color, linestyle='-', linewidth=width)
        ax.plot([pts[4][0], pts[5][0]], [pts[4][1], pts[5][1]], color=color, linestyle='-', linewidth=width)
        ax.plot([pts[4][0], pts[0][0]], [pts[4][1], pts[0][1]], color=color, linestyle='-', linewidth=width)
    if len(pts) == 4: # 2D points
        ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color=color, linestyle='-', linewidth=width)
        ax.plot([pts[1][0], pts[2][0]], [pts[1][1], pts[2][1]], color=color, linestyle='-', linewidth=width)
        ax.plot([pts[2][0], pts[3][0]], [pts[2][1], pts[3][1]], color=color, linestyle='-', linewidth=width)
        ax.plot([pts[3][0], pts[0][0]], [pts[3][1], pts[0][1]], color=color, linestyle='-', linewidth=width)


def save_image(args, ax, output_folder, frame):
    plot_ego(ax)
    plt.xlim(-75, 75)
    plt.ylim(-75, 75)
    image_path = os.path.join(output_folder, frame.replace("tiff", "png"))
    if args.debug: print("[Info] Saving", image_path)
    plt.savefig(image_path, format="png", dpi=1000, bbox_inches="tight")
    print("[INFO] Scene image saved at", image_path)
    plt.close()


def possible_overlap(args, line, closest, p1, p2, top, debug=False):
    det_closest = rectangle_find_closest(top)
    if Point(det_closest).distance_point([0, 0]) > args.dist_threshold_add + (args.dist_threshold_mult * Point(closest).distance_point([0, 0])):
        return False

    # Check if any of the detection corners lie between GT FOV
    v1 = Vector.from_points([0,0], p1)
    v2 = Vector.from_points([0,0], p2)
    a1to2 = v1.angle_signed(v2)
    if a1to2 > 0:
        min_a = 0
        max_a = a1to2
    else:
        min_a = a1to2
        max_a = 0

    for p in top:
        vp = Vector.from_points([0,0], p)
        a1top = v1.angle_signed(vp)
        if min_a <= a1top <= max_a: return True

    # Now flip the check and see if GT end points are in DET FOV
    vp0 = Vector.from_points([0,0], top[0])
    for p in top[1:]:
        a0top = vp0.angle_signed(p)
        if a0top > 0:
            min_a = 0
            max_a = a0top
        else:
            min_a = a0top
            max_a = 0
        av1 = vp0.angle_signed(v1)
        av2 = vp0.angle_signed(v2)
        if min_a <= av1 <= max_a: return True
        if min_a <= av2 <= max_a: return True

    return False


def merge_overlapping_projections(x_or_y, dets_projected):
    done = False
    found_overlap = False
    while(done == False):
        for i,det in enumerate(dets_projected):
            l1 = Line.from_points([0, 0], det[0])
            l2 = Line.from_points([0, 0], det[1])
            found_overlap = False
            for j,d in enumerate(dets_projected):
                if j <= i: continue
                if det[0][1] < 0 and det[1][1] < 0 and d[0][1] < 0 and d[1][1] < 0:
                    if (det[0][0] < 0 and det[1][0] > 0) or (det[0][0] > 0 and det[1][0] < 0) or (d[0][0] < 0 and d[1][0] > 0) or (d[0][0] > 0 and d[1][0] < 0):
                            found_overlap = True
                            break
                if l1.side_point(d[0]) == l2.side_point(d[0]) == l1.side_point(d[1]) == l2.side_point(d[1]): continue
                found_overlap = True
                break
            if found_overlap:
                break

        if found_overlap:
            d = dets_projected.pop(j)
            points = [d[0], d[1], dets_projected[i][0], dets_projected[i][1]]
            min_p = [9999, 9999]
            max_p = [-9999, -9999]
            for p in points:
                if p[x_or_y] > max_p[x_or_y]:
                    max_p = p
                if p[x_or_y] < min_p[x_or_y]:
                    min_p = p
            dets_projected[i] = [min_p, max_p]
        else:
            done = True

    return dets_projected


def do_one_gt(args, line, closest, p1, p2, dets, ax, segment=None, frame=None, index=None):

    # Use X or Y values to run rest of this analysis, prefer X
    if p1[0] == p2[0]:
        x_or_y = 1 # y
    elif p1[0] < p2[0] :
        x_or_y = 0 # x
    else:
        assert False, "p1[0] > p2[0] is unexpected due to prior sorting"

    # Converyt Dets to top view and break apart large detections
    tops = get_tops(args, dets, ax)

    # Plot top views of all detections
    # for t in tops:
    #     if ax: plot_one_box(ax, t, "red", width=1)

    dets_projected = []
    for top in tops:
        if not possible_overlap(args, line, closest, p1, p2, top, args.debug):
            continue # Remove tops based on distance and GT FOV
        if ax: plot_one_box(ax, top, "red", width=1)
        dp1, dp2 = get_projected_segment(line, top)
        assert(dp1[0] <= dp2[0])
        dets_projected.append([dp1, dp2])

    dets_projected = sorted(dets_projected, key = lambda x: x[0][x_or_y])
    # if debug: print (dets_projected)
    for det in dets_projected:
        if ax: ax.plot([det[0][0], det[1][0]], [det[0][1], det[1][1]], c='red', linewidth=2)

    dets_projected = merge_overlapping_projections(x_or_y, dets_projected)


    # Finally get around to finding overlap with GT projection
    total_len = Point(p1).distance_point(p2)
    covered_len = 0

    l1 = Line.from_points([0, 0], p1)
    l2 = Line.from_points([0, 0], p2)
    left = [l2.side_point(p1), 0] # P1's direction fom L2
    right = [l1.side_point(p2), 0] # P2's direction fom L1

    for det in dets_projected:
        p_min = np.array([None, None])
        p_max = np.array([None, None])

        # Ugh .. hacky
        # check for the special case, but unless all conditions are met fall though to the other if conditions.
        if p1[1] < 0 and p2[1] < 0 and det[0][1] < 0 and det[1][1] < 0:
            if (p1[0] < 0 and p2[0] > 0) or (p1[0] > 0 and p2[0] < 0) or (det[0][0] < 0 and det[1][0] > 0) or (det[0][0] > 0 and det[1][0] < 0):
                if p1[0] < det[0][0]: p_min = det[0]
                else: p_min = p1
                if p2[0] < det[1][0]: p_max = p2
                else: p_max = det[1]
                covered_len += Point(p_min).distance_point(p_max)
                continue


        # Line segment does not intersect with GT, should not be happening by this point but still check
        if l1.side_point(det[0]) in left and l1.side_point(det[1]) in left:
            assert False

        if l2.side_point(det[0]) in right and l2.side_point(det[1]) in right:
            assert False


        if l1.side_point(det[0]) in left and l2.side_point(det[1]) in left:
            p_min = p1
            p_max = det[1]

        if l1.side_point(det[0]) in right and l2.side_point(det[1]) in right:
            p_min = det[0]
            p_max = p2

        if l1.side_point(det[0]) in right and l2.side_point(det[1]) in left:
            p_min = det[0]
            p_max = det[1]

        if l1.side_point(det[0]) in left and l2.side_point(det[1]) in right:
            p_min = p1
            p_max = p2

        if args.debug: print("GT Index:", index)
        if args.debug: print(p1, p2)
        if args.debug: print(left, right)
        if args.debug: print(det[0], det[1])
        if args.debug: print(p_min, p_max)
        if args.debug: print(l1.side_point(det[0]), l2.side_point(det[1]))

        if set(left) == set(right):
            assert False, "[Error] GT Sides are same, bug"

        if p_min.any() == None or p_max.any() == None:
            print ("[Error]", segment, frame)
            covered_len += 0
            if ax: ax.plot([det[0][0], det[1][0]], [det[0][1], det[1][1]], c='black', linewidth=2)
            assert False, "[Error] This is a bug"
        else:
            covered_len += Point(p_min).distance_point(p_max)

        if args.debug: print(covered_len*100/total_len)
        if args.debug: print("\n")

    return covered_len/total_len


def do_one_frame(args, gt, dt, f, output_folder, segment=None, frame=None):
    gt_out = []

    if args.save_images: ax = plt.subplot()
    else: ax = None

    for i,g in enumerate(gt):

        if args.gt_index != None:
            if i != args.gt_index:
                continue
        if args.gt_id != None:
            if g[GT_ID_INDEX] != args.gt_id:
                continue


        pts = get_3d_bbox_egovehicle_frame(g)
        top = top_from_box(pts)
        p = rectangle_find_closest(top)
        slope = -1 * p[0] / p[1]
        line = Line.from_slope(slope, p[1] - slope * p[0])
        p1, p2 = get_projected_segment(line, top)

        if ax: plot_one_box(ax, pts, 'green')
        if ax: plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='darkgreen', linewidth=4)

        covered_ratio = do_one_gt(args, line, p, p1, p2, dt, ax, segment, frame, i)
        g.append(covered_ratio)
        gt_out.append(g)
        if ax: plt.text(p[0] , p[1]+ 5, str(int(covered_ratio*100)) + "%", size="small")


    if ax:
        save_image(args, ax, output_folder, f)

    return gt_out


def do_one_segment(segment_path, args, tqdm_position):

    signal.signal(signal.SIGINT, handle_sigint)

    if args.debug: print("Starting segment", segment_path.split("/")[-1])
    with open(os.path.join(segment_path, args.gt_name), 'r') as gtf:
        groundtruths = json.load(gtf)
    with open(os.path.join(segment_path, "frames_lidar/first_return", args.det_name), 'r') as dtf:
        detections = json.load(dtf)

    with tqdm_position.get_lock():
        tqdm_position.value += 1
        pbar = tqdm.tqdm(total=len(groundtruths.keys()), position=tqdm_position.value, leave=False)
        pbar.set_description(str(tqdm_position.value).rjust(4, " ") + " : " + segment_path.split("/")[-1].ljust(70, " "))

    gt_dict = dict()

    output_folder = os.path.join(segment_path, OUTPUT_DIR)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for i,frame in enumerate(groundtruths.keys()):
        if args.frame:
            if args.frame != i:
                continue
        gt_list = do_one_frame(args, groundtruths[frame], detections[frame], frame, output_folder, segment_path, frame)
        gt_dict[frame] = gt_list
        pbar.update(1)

    outfile = os.path.join(output_folder, args.outfile)
    with open(outfile, "w") as gtf:
        json.dump(gt_dict, gtf, indent = 4)

    pbar.close()
    return outfile

def do_one_segment_wrapper(segment_path):
    global args, tqdm_position
    return do_one_segment(segment_path, args, tqdm_position)

def main():
    global args
    args = get_args() # We will just use the global args

    if args.segment:
        segment_paths = [os.path.realpath(args.segment)]
    else:
        segment_paths = ([os.path.join(args.dataset, f) for f in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, f))])


    if args.skip_existing:
        segments_without_output = []
        for segment_path in segment_paths:
            if not os.path.exists(os.path.join(segment_path, OUTPUT_DIR, args.outfile)):
                segments_without_output.append(segment_path)
            else:
                print("[Info] Skipping segment", segment_path, "due to existing output.")
        segment_paths = segments_without_output

    print("[Info] Evaluating", len(segment_paths), "scenes")
    with multiprocessing.Pool(processes = args.threads) as pool:
        if len(segment_paths) > args.threads:
            pool_chunksize = len(segment_paths) // args.threads
        else: # Small batch version for test runs
            pool_chunksize = 1
        res = pool.imap_unordered(do_one_segment_wrapper, segment_paths, pool_chunksize)
        res = list(res)


if __name__ == '__main__':
    main()
