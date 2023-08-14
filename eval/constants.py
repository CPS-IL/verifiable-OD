from waymo_open_dataset import label_pb2

# Ego vehicle dimenstion
ego_length = 5.1816
ego_width = 2.032

rec_ego = [
    [ ego_length/2,  ego_width/2],
    [-ego_length/2,  ego_width/2],
    [ ego_length/2, -ego_width/2],
    [-ego_length/2, -ego_width/2],
]

max_acceleration_lookup = {
    label_pb2.Label.Type.TYPE_UNKNOWN    : 7.5,
    label_pb2.Label.Type.TYPE_VEHICLE    : 7.5,
    label_pb2.Label.Type.TYPE_PEDESTRIAN : 7.5,
    label_pb2.Label.Type.TYPE_SIGN       : 0,
    label_pb2.Label.Type.TYPE_CYCLIST    : 4, # https://bicycles.stackexchange.com/questions/75262/how-fast-can-a-cyclist-accelerate-from-a-standing-start
}

lidar_range = 75