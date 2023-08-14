
import os
import cv2
import argparse
import numpy as np
import json
import math
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import frame_utils

import library
from constants import lidar_range

def get_filepaths(args):
    tfrecords = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(args.dataset)) for f in fn]
    file_paths = []
    for f in tfrecords:
        if not os.path.isfile(f):
            continue
        if not "tfrecord" in f:
            continue
        file_paths.append(f)
    return file_paths


def generate_logger_configuration(configuration_tree):
    logger_tree = {}
    logger_tree["log_path"] = "./"
    logger_tree["log_file_name_cube"] = "depth_clustering_detection_cube.json"
    logger_tree["log_file_name_polygon"] = "depth_clustering_detection_polygon.json"
    logger_tree["log_file_name_flat"] = "depth_clustering_detection_flat.json"
    logger_tree["log"] = True
    configuration_tree["logger"] = logger_tree
    return


def generate_depth_clustering_configuration(configuration_tree, dataset_path):
    depth_clustering_tree = {}
    depth_clustering_tree["distance_clustering"] = 5
    depth_clustering_tree["score_clustering"] = 1
    depth_clustering_tree["angle_clustering"] = 10
    depth_clustering_tree["angle_ground_removal"] = 9
    depth_clustering_tree["size_cluster_min"] = 10
    depth_clustering_tree["size_cluster_max"] = 20000
    depth_clustering_tree["size_smooth_window"] = 5
    depth_clustering_tree["use_camera_fov"] = True
    depth_clustering_tree["use_score_filter"] = False
    depth_clustering_tree["score_filter_threshold"] = 0.5
    depth_clustering_tree["score_type_point"] = "type_1"
    depth_clustering_tree["score_type_cluster"] = "type_1"
    depth_clustering_tree["score_type_frame"] = "type_1"
    depth_clustering_tree["bounding_box_type"] = "polygon"
    depth_clustering_tree["difference_type"] = "angles_precomputed"
    depth_clustering_tree["dataset_file_type"] = ".tiff"
    depth_clustering_tree["dataset_name"] = dataset_path.split("/")[-1].split(".tfrecord")[0]
    depth_clustering_tree["ground_truth_cube_file_name"] = "waymo_ground_truth_cube.json"
    depth_clustering_tree["ground_truth_flat_file_name"] = "waymo_ground_truth_flat.json"
    configuration_tree["depth_clustering"] = depth_clustering_tree
    return


def generate_lidar_projection_configuration(configuration_tree, frame):
    lidar_projection_tree = {}

    lidar_frames, _, _ = frame_utils.parse_range_image_and_camera_projection(frame)
    range_image_lidar_top = lidar_frames[dataset_pb2.LaserName.TOP][0]
    range_image_tensor = tf.convert_to_tensor(range_image_lidar_top.data)
    range_image_tensor = tf.reshape(range_image_tensor, range_image_lidar_top.shape.dims)
    lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
    range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                            tf.ones_like(range_image_tensor) * 1e10)
    range_image_range = range_image_tensor[...,0]
    horizontal_steps = range_image_range.shape[1]

    # Obtain lidar calibrations
    lidar_calibrations = frame.context.laser_calibrations
    beam_inclinations = []
    extrinsic = []

    # Obtain top lidar beam inclinations
    for lidar_calibration in lidar_calibrations:
        if lidar_calibration.name == dataset_pb2.LaserName.TOP:
            beam_inclinations = list(lidar_calibration.beam_inclinations)
            extrinsic = list(lidar_calibration.extrinsic.transform)
            break

    # Reverse inclinations
    beam_inclinations.reverse()

    # Write to lidar projection configuration tree
    lidar_projection_tree["horizontal_steps"] = horizontal_steps
    lidar_projection_tree["beams"] = len(beam_inclinations)
    lidar_projection_tree["horizontal_angle_start"] = 180
    lidar_projection_tree["horizontal_angle_end"] = -180
    lidar_projection_tree["intensity_norm_factor"] = 0.71
    lidar_projection_tree["elongation_norm_factor"] = 1.5
    lidar_projection_tree["beam_inclinations"] = np.degrees(beam_inclinations).tolist()
    lidar_projection_tree["extrinsic"] = extrinsic

    configuration_tree["lidar_projection"] = lidar_projection_tree
    return


def generate_camera_projection_configuration(configuration_tree, frame):
    camera_projection_tree = {}

    camera_calibrations = frame.context.camera_calibrations
    intrinsic = []
    extrinsic = []
    width = 0
    height = 0

    # Obtain front lidar beam inclinations
    for camera_calibration in camera_calibrations:
        if camera_calibration.name == dataset_pb2.CameraName.FRONT:
            intrinsic = list(camera_calibration.intrinsic)
            extrinsic = list(camera_calibration.extrinsic.transform)
            width = camera_calibration.width
            height = camera_calibration.height
            break

    camera_projection_tree["intrinsic"] = intrinsic
    camera_projection_tree["extrinsic"] = extrinsic
    camera_projection_tree["width"] = width
    camera_projection_tree["height"] = height
    camera_projection_tree["field_of_view_angle_start"] = 26
    camera_projection_tree["field_of_view_angle_end"] = -26
    camera_projection_tree["threshold_truncation"] = 1
    camera_projection_tree["threshold_filter_height"] = 0
    camera_projection_tree["threshold_filter_tunnel_left"] = 1000
    camera_projection_tree["threshold_filter_tunnel_right"] = 1000
    camera_projection_tree["threshold_filter_tunnel_front"] = 75
    camera_projection_tree["correct_distortions"] = False
    camera_projection_tree["use_filter_height"] = False
    camera_projection_tree["use_filter_tunnel"] = False

    configuration_tree["camera_projection"] = camera_projection_tree

    return


def generate_configuration(args, dataset_path, frame):
    # Define variables
    configuration_tree = {}
    configuration_file_name = "depth_clustering_config.json"

    # Generate configurations
    generate_depth_clustering_configuration(configuration_tree, dataset_path)
    generate_lidar_projection_configuration(configuration_tree, frame)
    generate_camera_projection_configuration(configuration_tree, frame)
    generate_logger_configuration(configuration_tree)

    # Define output strings
    output_folder = dataset_path.split(".tfrecord")[0] + "/"
    output_print = "dataset"

    # Create new output data folder if not exist
    if os.path.exists(output_folder) == False:
        os.mkdir(output_folder)

    with open(output_folder + configuration_file_name , "w") as output_file:
        json.dump(configuration_tree, output_file, indent = 4)

    if args.verbose:
        print("[INFO]: Generated " + output_print + " configuration file \"" + output_folder + configuration_file_name + "\".")

    return


# return true if label needs to be included,
def include_gt(ego_velocity, label, output_folder=None, exclude_before_first_beam=False, exclude_after_range=False, exclude_no_risk=False):

    if label.type == label_pb2.Label.Type.TYPE_SIGN:
        return False # We only consider obstacles on the road

    if exclude_before_first_beam:
        _, _, closest_depth_from_center = library.get_bounding_box_depth(label)

        # Beam inclinations are from Lidar sensor or origin
        config_file = os.path.join(output_folder, "depth_clustering_config.json")
        assert os.path.exists(config_file), "[ERROR]: Config file missing, should have been generated already in this script."
        with open(config_file) as c:
            cfg = json.load(c)

        beam_inclinations = cfg['lidar_projection']['beam_inclinations']
        beam_inclination = beam_inclinations[len(beam_inclinations) - 1]
        lidar_height = cfg['lidar_projection']['extrinsic'][11]
        dist_limit_min = abs(lidar_height/math.tan(math.radians(abs(beam_inclination))))
        if closest_depth_from_center <= dist_limit_min:
            return False # Obstacle closer than first beam on ground, see paper, constraint C3

    if exclude_after_range and library.get_depth_center_to_center(label) >= lidar_range:
        return False # Max range of waymo lidar

    if exclude_no_risk:
        ego_pos = (0, 0) # Ego vehicle is origin
        collision_category = library.getCollisionPossibility(ego_pos, ego_velocity, label)
        if collision_category == "Potential" or collision_category == "Imminent":
            return True
        else:
            return False # No Risk of Collision

    return True


def get_translational_vector(frame):
    lidar_calibrations = frame.context.laser_calibrations

    for lidar_calibration in lidar_calibrations:
        if lidar_calibration.name == dataset_pb2.LaserName.TOP:
            extrinsic = list(lidar_calibration.extrinsic.transform)
            break

    extrinsic_translational_vector_x = extrinsic[3]
    extrinsic_translational_vector_y = extrinsic[7]
    extrinsic_translational_vector_z = extrinsic[11]

    return (extrinsic_translational_vector_x, extrinsic_translational_vector_y, extrinsic_translational_vector_z)


def label_to_list_cube(label, velocity, translational_vector):
    ground_truth_box_cube = []
    ground_truth_box_cube.append(label.box.center_x - translational_vector[0])
    ground_truth_box_cube.append(label.box.center_y - translational_vector[1])
    ground_truth_box_cube.append(label.box.center_z - translational_vector[2])
    ground_truth_box_cube.append(label.box.length)
    ground_truth_box_cube.append(label.box.width)
    ground_truth_box_cube.append(label.box.height)
    ground_truth_box_cube.append(label.box.heading)
    ground_truth_box_cube.append(label.id)
    ground_truth_box_cube.append(label.metadata.speed_x)
    ground_truth_box_cube.append(label.metadata.speed_y)
    ground_truth_box_cube.append(velocity[0])
    ground_truth_box_cube.append(velocity[1])
    ground_truth_box_cube.append(label.type)
    return ground_truth_box_cube


def extract_ground_truth_cube(
        args, frame, index, ground_truths_tree, output_folder, translational_vector,
        exclude_before_first_beam, exclude_after_range, exclude_no_risk):
    '''Extract the ground truths into JSON tree.'''
    ground_truth_boxes_cube = []

    velocity_vehicle_x = 0
    velocity_vehicle_y = 0
    for image in frame.images:
        velocity_vehicle_x += image.velocity.v_x
        velocity_vehicle_y += image.velocity.v_y
    velocity_vehicle_x /= len(frame.images)
    velocity_vehicle_y /= len(frame.images)

    for ground_truth_cube in frame.laser_labels:

        if not include_gt((velocity_vehicle_x, velocity_vehicle_y), ground_truth_cube, output_folder,
                          exclude_before_first_beam, exclude_after_range, exclude_no_risk):
            continue

        ground_truth_boxes_cube.append(label_to_list_cube(ground_truth_cube, (velocity_vehicle_x, velocity_vehicle_y), translational_vector))

    ground_truths_tree["frame_lidar_first_return_range_" + str(index) + ".tiff"] = ground_truth_boxes_cube

    if args.verbose:
        print("[INFO]: Extracted cube ground truths for frame " + str(index) + ".")

    return


def extract_lidar_frame(args, dataset_path, lidar_frames, frame_number):
    # Define variables

    lidar_return = 0 # First return only
    lidar_channel = 0 # Range image

    # Compose output names
    output_path = dataset_path.split(".tfrecord")[0] + "/frames_lidar/first_return/range/"
    output_file_name_prefix = "frame_lidar_first_return_range_"

    # Create output path if nonexistent
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    output_file_path = output_path + output_file_name_prefix + str(frame_number) + ".tiff"

    # Need to extract, do the work of frame extraction now
    lidar_frame = lidar_frames[dataset_pb2.LaserName.TOP][lidar_return]
    lidar_frame_tensor = tf.convert_to_tensor(lidar_frame.data)
    lidar_frame_tensor = tf.reshape(lidar_frame_tensor, lidar_frame.shape.dims)
    lidar_image_mask = tf.greater_equal(lidar_frame_tensor, 0)
    lidar_frame_tensor = tf.where(lidar_image_mask, lidar_frame_tensor,
                                tf.ones_like(lidar_frame_tensor) * 1e10)

    lidar_frame_channel = lidar_frame_tensor[..., lidar_channel]


    # Set invalid range values to zero
    lidar_frame_channel = np.where(lidar_frame_channel >= 10000000000.0, 0.0, lidar_frame_channel)

    # Convert to 16-bit values
    lidar_frame_channel /= 75.0
    lidar_frame_channel *= 65535.0
    lidar_frame_channel = np.where(lidar_frame_channel > 65535.0, 65535.0, lidar_frame_channel)
    lidar_frame_channel = lidar_frame_channel.astype(np.uint16)

    # Write frame to file
    cv2.imwrite(output_file_path, lidar_frame_channel)
    if args.verbose:
        print("[INFO]: Extracted lidar frame " + str(frame_number) + ".")

    return


def extract_segment_data(args, segment, tfrecord_path):
    for data in segment:
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        break

    generate_configuration(args, tfrecord_path, frame)
    translational_vector = get_translational_vector(frame)

    return translational_vector


def write_gt_file(args, output_folder, filename, gt_tree):
    gt_file_path = os.path.join(output_folder, filename)
    with open(gt_file_path, "w") as output_file:
       json.dump(gt_tree, output_file, indent = 4)
    if args.verbose:
        print("[INFO]: Generated ground truth file " + gt_file_path)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset", required=True)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    tfrecord_paths = get_filepaths(args)
    for tfrecord_path in tfrecord_paths:
        ground_truths_tree_safety_critical = {}

        output_folder = tfrecord_path.split(".tfrecord")[0] + "/"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        segment = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
        translational_vector = extract_segment_data(args, segment, tfrecord_path)

        for frame_number, data in enumerate(segment):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            extract_ground_truth_cube(args, frame, frame_number, ground_truths_tree_safety_critical, output_folder, translational_vector,
                                      exclude_before_first_beam=True, exclude_after_range=True, exclude_no_risk=True)

            lidar_frames, _, _ = frame_utils.parse_range_image_and_camera_projection(frame)
            extract_lidar_frame(args, tfrecord_path, lidar_frames, frame_number)

            camera_outdir = os.path.join(output_folder, "frames_camera")
            if not os.path.exists(camera_outdir):
                os.mkdir(camera_outdir)
            plt.imsave(os.path.join(camera_outdir, "frame_camera_" + str(frame_number) + ".png"),
                np.array(tf.image.decode_jpeg(frame.images[0].image)))

            if args.verbose:
                print("[INFO]: Extracted camera frame " + str(frame_number) + ".")

        write_gt_file(args, output_folder, "waymo_ground_truth_cube_safety.json", ground_truths_tree_safety_critical)

        print("[INFO]: Extracted all data for", tfrecord_path, "\n")


if __name__ == "__main__":
    main()
