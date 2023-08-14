import math
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation
from skspatial.objects import Line
from waymo_open_dataset import label_pb2
from waymo_open_dataset import dataset_pb2
from constants import ego_length, ego_width, max_acceleration_lookup



def get_bounding_box_corners(label):
    center_x = label.box.center_x
    center_y = label.box.center_y
    center_z = label.box.center_z
    extent_x = label.box.length
    extent_y = label.box.width
    extent_z = label.box.height

    corners_x = np.array([
        center_x - extent_x / 2,
        center_x + extent_x / 2,
        center_x - extent_x / 2,
        center_x + extent_x / 2,
        center_x - extent_x / 2,
        center_x + extent_x / 2,
        center_x - extent_x / 2,
        center_x + extent_x / 2
    ])

    corners_y = np.array([
        center_y - extent_y / 2,
        center_y - extent_y / 2,
        center_y + extent_y / 2,
        center_y + extent_y / 2,
        center_y - extent_y / 2,
        center_y - extent_y / 2,
        center_y + extent_y / 2,
        center_y + extent_y / 2
    ])

    corners_z = np.array([
        center_z - extent_z / 2,
        center_z - extent_z / 2,
        center_z - extent_z / 2,
        center_z - extent_z / 2,
        center_z + extent_z / 2,
        center_z + extent_z / 2,
        center_z + extent_z / 2,
        center_z + extent_z / 2
    ])

    return (corners_x, corners_y, corners_z)

def get_bounding_box_depth_old(bounding_box):
    bounding_box_corners = get_bounding_box_corners(bounding_box)
    corners_x = bounding_box_corners[0]
    corners_y = bounding_box_corners[1]
    corners_z = bounding_box_corners[2]

    depth = math.sqrt((corners_x[0] ** 2) + (corners_y[0] ** 2) + (corners_z[0] ** 2))
    corner = (corners_x[0], corners_y[0], corners_z[0])

    for i in range(1,8):
        current_depth = math.sqrt((corners_x[i] ** 2) + (corners_y[i] ** 2) + (corners_z[i] ** 2))

        if current_depth < depth:
            depth = current_depth
            corner = (corners_x[i], corners_y[i], corners_z[i])

    return (depth, corner)


def top_from_box(pts):
    return [
        [pts[0][0], pts[0][1]],
        [pts[1][0], pts[1][1]],
        [pts[5][0], pts[5][1]],
        [pts[4][0], pts[4][1]],
    ]

# Source https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
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

    return [reqAns, reqP]

def rectangle_find_closest(rec):
    rec_ego = [
        [ ego_length/2,  ego_width/2],
        [-ego_length/2,  ego_width/2],
        [ ego_length/2, -ego_width/2],
        [-ego_length/2, -ego_width/2],
    ]

    ego_min = []
    rec_min = []
    for i,_ in enumerate(rec_ego):
        for j,_ in enumerate(rec):
            if j == len(rec) - 1: j_next = 0
            else: j_next = j + 1
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
        return min_rec[1], min_rec[0]
    else:
        rec_min = []
        for j,_ in enumerate(rec):
            if j == len(rec) - 1: j_next = 0
            else: j_next = j + 1
            rec_min.append(minDistance(rec[j], rec[j_next], min_ego[1]))
        for e in rec_min:
            if e[0] < min_rec[0]:
                min_rec = e
        return min_rec[1], min_rec[0]

# Source https://github.com/waymo-research/waymo-open-dataset/issues/192
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

def get_3d_bbox_egovehicle_frame(label):
	# 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
	x_corners = label.box.length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
	y_corners = label.box.width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
	z_corners = label.box.height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
	corners_obj_frame = np.vstack((x_corners, y_corners, z_corners)).T

	ego_T_obj = np.array([ label.box.center_x, label.box.center_y, label.box.center_z ])
	egovehicle_SE3_object = SE3_from_R_t(
		rotation=heading_to_rotmat(label.box.heading),
		translation=ego_T_obj,
	)
	egovehicle_pts = apply_SE3(egovehicle_SE3_object, corners_obj_frame)
	return egovehicle_pts

def get_bounding_box_depth(label):
    bounding_box_corners = get_3d_bbox_egovehicle_frame(label)
    bounding_box_top_corners = top_from_box(bounding_box_corners)
    closest_point_to_egovehicle, closest_depth = rectangle_find_closest(bounding_box_top_corners)
    closest_depth_from_center = math.sqrt(math.pow(closest_point_to_egovehicle[0], 2) + math.pow(closest_point_to_egovehicle[1], 2))
    return (closest_depth, closest_point_to_egovehicle, closest_depth_from_center)

def get_depth_center_to_center(label):
    return math.sqrt(math.pow(label.box.center_x, 2) + math.pow(label.box.center_y, 2) + math.pow(label.box.center_z, 2))

# def filter_bounding_box_tunnel(bounding_box, tunnel_margin_left = 1000,
#                                 tunnel_margin_right = 1000, tunnel_margin_front = 75):
#     depth, _ = get_bounding_box_depth(bounding_box)
#
#     return (bounding_box[1] <= tunnel_margin_right and
#             bounding_box[1] >= -tunnel_margin_left and
#             depth <= tunnel_margin_front)

def filter_bounding_box_truncation(label, truncation_angle_threshold = 26):
    if truncation_angle_threshold == 0: return True # A disable filter condition

    _, corner = get_bounding_box_depth_old(label)
    truncation_angle = abs(math.degrees(math.tan(corner[1] / corner[0])))

    return truncation_angle < truncation_angle_threshold

def createTrajectory(x, v, timestep, iterations):
    trajectory = {}
    trajectory[0] = (x[0], x[1])
    for i in range(1, iterations):
        trajectory[timestep * i] = (trajectory[0][0] + (v[0] * timestep * i),
                                    trajectory[0][1] + (v[1] * timestep * i))
    return trajectory

def getDist(a, b):
    return math.sqrt(pow(a[0] - b[0], 2) +
                     pow(a[1] - b[1], 2))

def getIntersect3d(label, ego_center, obj_center):
    ego_2d = [[ 2 + ego_center[0],  1 + ego_center[1]],
              [ 2 + ego_center[0], -1 + ego_center[1]],
              [-2 + ego_center[0], -1 + ego_center[1]],
              [-2 + ego_center[0],  1 + ego_center[1]]]
    box = label.box
    cx = obj_center[0]
    cy = obj_center[1]
    cz = box.center_z
    l = box.length
    w = box.width
    h = box.height
    ry = box.heading
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [w, -w, -w, w, w, -w, -w, w]
    z_corners = [h, h, h, h, -h, -h, -h, -h]
    R = np.array([[np.cos(ry), -np.sin(ry), 0], [np.sin(ry), np.cos(ry), 0], [0, 0, 1]])
    corners3d = np.vstack([x_corners, y_corners, z_corners]) / 2.0
    corners3d = (R @ corners3d).T + np.array([cx, cy, cz])
    p1 = Polygon([
        (corners3d[0][0], corners3d[0][1]),
        (corners3d[1][0], corners3d[1][1]),
        (corners3d[2][0], corners3d[2][1]),
        (corners3d[3][0], corners3d[3][1])])
    p2 = Polygon([
        (ego_2d[0][0], ego_2d[0][1]),
        (ego_2d[1][0], ego_2d[1][1]),
        (ego_2d[2][0], ego_2d[2][1]),
        (ego_2d[3][0], ego_2d[3][1])])
    return(p1.intersects(p2))

def getCollisionPossibility(ego_x, ego_v, label, a_max=None, t_compute=0.1, timestep=0.1):
    if a_max == None: a_max = max_acceleration_lookup[label.type]
    a_max_ego = max_acceleration_lookup[label_pb2.Label.Type.TYPE_VEHICLE]
    obj_x = (label.box.center_x, label.box.center_y)
    obj_v = (label.metadata.speed_x, label.metadata.speed_y)
    TTS = t_compute + math.sqrt(pow(ego_v[0], 2) + pow(ego_v[1], 2))/a_max_ego
    iterations = int(TTS/timestep) + 1 # number of iterations

    ego_traj = createTrajectory(ego_x, ego_v, timestep, iterations)
    obj_traj = createTrajectory(obj_x, obj_v, timestep, iterations)
    dist_crit = math.sqrt(pow(label.box.length, 2) + pow(label.box.width, 2)) + (math.sqrt(pow(ego_length, 2) + pow(ego_width, 2))/2)

    for i in range(iterations):
        if getIntersect3d(label, ego_traj[timestep * i], obj_traj[timestep * i]):
            return "Imminent"
        if getDist(ego_traj[timestep * i], obj_traj[timestep * i]) < \
                        (dist_crit + 0.5 * a_max * pow(timestep * i, 2) + 0.5 * a_max_ego * pow(timestep * i, 2)):
            return "Potential"
    return "None"


class Bb2d:
    def __init__(self, x1, y1, x2, y2, cat = None, conf = 1, id = None, dist = 0, velocity = (0, 0)):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.cat = cat
        self.conf = conf
        self.id = id
        self.dist = dist
        self.velocity = velocity

    def __str__(self):
        return str(self.x1) + ' '+str(self.y1) + ' '+str(self.x2) + ' '+str(self.y2) + ', cat:'+str(self.cat) + ', conf:'+str(self.conf)

    def plot(self, ax, c='black'):
        rect = patches.Rectangle((self.x1, self.y1), self.x2 - self.x1, self.y2 - self.y1,
            linewidth = 2, edgecolor = c, facecolor = 'none')
        ax.add_patch(rect)

    def get_areas(self, boxB):
        boxA = self
        xA = max(boxA.x1, boxB.x1)
        yA = max(boxA.y1, boxB.y1)
        xB = min(boxA.x2, boxB.x2)
        yB = min(boxA.y2, boxB.y2)
        interArea = max(0, xB - xA ) * max(0, yB - yA )
        boxAArea = (boxA.x2 - boxA.x1 ) * (boxA.y2 - boxA.y1 )
        boxBArea = (boxB.x2 - boxB.x1 ) * (boxB.y2 - boxB.y1 )
        return boxAArea, boxBArea, interArea

    def intersection_over_union(self, boxB):
        boxAArea, boxBArea, interArea = self.get_areas(boxB)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def intersection_over_self_area(self, boxB):
        boxAArea, _, interArea = self.get_areas(boxB)
        iog = interArea / float(boxAArea)
        return iog


def waymo_label_to_BB(label):
    return Bb2d(float(label.box.center_x - 0.5 * label.box.length),
                float(label.box.center_y - 0.5 * label.box.width),
                float(label.box.center_x + 0.5 * label.box.length),
                float(label.box.center_y + 0.5 * label.box.width),
                int(label.type), id=label.id)


def simons_waymo_flat_label_to_BB(label):
    return Bb2d(float(label[0]), float(label[1]), float(label[2]), float(label[3]),
                    cat = label[10],
                    conf = 1,
                    id = label[5],
                    dist = float(label[4]),
                    velocity = (float(label[6]), float(label[7])))


# Given the groundtruth bbx and one predicted bb, it returns the maximum iou
# obtained and the index of the gt bbx
def find_best_match(target_bbs, bb, iog=False):
    iou = []
    for tbb in target_bbs:
        if iog:
            iou.append(bb.intersection_over_self_area(tbb))
        else:
            iou.append(bb.intersection_over_union(tbb))

    if iou == []:
        return 0, -1

    iou_max = max(iou)
    i = iou.index(iou_max)
    return iou_max, i


# Find best matching projected 3D label for 2D label.
def waymo_find_best_match_id(frame, camera_label, iog=False, iou_thresh=0.5):
    target_bbs = []
    for label in frame.projected_lidar_labels[0].labels:
        target_bbs.append(waymo_label_to_BB(label))
    bb = waymo_label_to_BB(camera_label)
    iou_max, i = find_best_match(target_bbs, bb, iog)
    if iou_max >= iou_thresh:
        return target_bbs[i].id.split('_')[0]
    else:
        return None
