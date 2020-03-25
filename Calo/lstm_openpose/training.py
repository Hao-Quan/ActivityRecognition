import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
import shutil
import xml.etree.ElementTree as ET

from collections import Counter
from datetime import datetime
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from xml.etree.ElementTree import Element, ElementTree

#import pose_estimation.build.human_pose_estimation_demo.python.chpe as chpe

# -----------------------------------------------------
# RNN-for-Human-Activity-Recognition-using-2D-Pose-Input VARIABLES
# -----------------------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0
from sklearn import metrics
import random
import time
# -----------------------------------------------------


# -----------------------------------------------------
# DEBUGGING VARIABLES
# -----------------------------------------------------
N_FRAMES_LIMITATION = 50
LIMIT_FRAME = False
DEBUG_FRAME = -1
DEBUG_PRINTS = False
STATUS_FRAME_RATE = 50
# -----------------------------------------------------


# -----------------------------------------------------
# GENERAL VARIABLES
# -----------------------------------------------------

config = {}

# -----------------------------------------------------
# PATH VARIABLES
# -----------------------------------------------------
CVAT_DUMPS_PATH = 'video_tag/cvat_annotations'
# ANNOTATIONS_PATH = 'video_tag/cvat_annotations_sorted'
# VIDEOS_PATH = 'video_tag/video'
# HAR_DATASET_PATH = 'activity_recognition/data'
MODEL_DIR = 'activity_recognition/model'
MODEL_NAME = 'model'
MODEL_FILE_NAME = MODEL_NAME + '.meta'
CLASS_MAP_FILE = 'class_map.json'

# X_TRAIN = 'X_train.txt'
# X_TEST = 'X_test.txt'
# Y_TRAIN = 'Y_train.txt'
# Y_TEST = 'Y_test.txt'
X_DATA = 'X.txt'
Y_DATA = 'Y.txt'
# -----------------------------------------------------


# -----------------------------------------------------
# TRAINING VARIABLES
# -----------------------------------------------------
MAX_HEIGHT = 100
MAX_WIDTH = 100

# labels = []     # built at run-time depending on CVAT labels

# DATASET_PATH = os.path.join(HAR_DATASET_PATH, '1_carosello')
# X_train_path = os.path.join(DATASET_PATH, X_TRAIN)
# X_test_path = DATASET_PATH + "X_test.txt"
# Y_train_path = os.path.join(DATASET_PATH, Y_TRAIN)
# y_test_path = DATASET_PATH + "Y_test.txt"

n_steps = 32  # 32 timesteps per series
n_joints = 18  # COCO notation
# -----------------------------------------------------

# -----------------------------------------------------
# PLOT VARIABLES
# -----------------------------------------------------
MAX_Y_CONF_MATRIX = 50
# -----------------------------------------------------


class MetaData(object):
    video = ""
    tag = ""
    cvat_dump = ""
    skeletons = ""

    def __init__(self, video, tag, cvat_dump, skeletons):
        self.video = video
        self.tag = tag
        self.cvat_dump = cvat_dump
        self.skeletons = skeletons


def get_default_tf_session_path():
    return os.path.join('activity_recognition/model', 'model')


def initialise_metadata(from_files=None):
    if from_files is None:
        from_files = []

    data = []
    # print("from_files arg: ", from_files)
    # print("type(from_files): ", type(from_files))
    if from_files:
        files = [os.path.join(config['cvat_dumps_path'], f) for f in from_files
                 if is_xml_file(config['cvat_dumps_path'], f)]
    else:
        files = [os.path.join(config['cvat_dumps_path'], f) for f in list_file_in(config['cvat_dumps_path'])
                 if is_xml_file(config['cvat_dumps_path'], f)]

    print("Raw CVAT dumps found: ", files)

    if not files:
        raise ValueError('No CVAT dump found!')

    for file in files:
        base_file = os.path.basename(file)
        # print("Found: " + base_file)

        cvat_dump = file

        # Compute path of sorted cvat , i.e. file that will contain tags
        tag_file = os.path.join(config['tags_path'], base_file)

        # Compute corresponding video path
        no_ext_video_file = os.path.splitext(base_file)[0]
        video_file = os.path.join(config['video_path'], no_ext_video_file + ".m4v")

        # Creating output folder
        skeleton_path = os.path.join(config['har_dataset_path'], no_ext_video_file)
        if not os.path.exists(skeleton_path):
            os.makedirs(skeleton_path)

        input = MetaData(video_file, tag_file, cvat_dump, skeleton_path)

        data.append(input)

    data.reverse()

    return data


def extract_class_map(tag_file):
    # Initialise class_map
    classes = extract_classes(tag_file)
    class_id = 0
    loc_class_map = {}
    for cl in classes:
        loc_class_map[cl] = class_id
        class_id += 1
    return loc_class_map


def print_xml(elem):
    xml_str = ET.tostring(elem, encoding='unicode', method='xml')
    print(xml_str)


def get_first_frame(track_el):
    return int(track_el.find("./box[1]").attrib["frame"])


def get_last_frame(track_el):
    return int(track_el.find("./box[last()]").attrib["frame"])


def list_file_in(path):
    return os.listdir(path)


def is_xml_file(path, f):
    return os.path.isfile(os.path.join(path, f)) and '.xml' in f


def extract_boxes_present_in(root, frame_id):

    boxes = root.findall("./track/box[@frame='"+str(frame_id)+"']")

    return boxes


def extract_classes(annotation_file_path):
    tree = ET.parse(annotation_file_path)
    root = tree.getroot()

    class_els = root.findall(".//labels/label/name")

    classes = []
    for class_el in class_els:
        if "together" not in class_el.text:
            rename = class_el.text
            if class_el.text == 'walking_caddy':
                rename = 'walking_cart'
            elif class_el.text == 'standing_busy':
                rename = 'standing_phone_talking'
            elif class_el.text == 'standing_free':
                rename = 'standing'
            classes.append(rename)

    return classes


def extract_beginning_track(root, frame_id):
    print('extract_beginning_track()')

    beg_traces = []

    traces = root.findall("./track")

    for track in traces:
        first_frame = get_first_frame(track)
        if first_frame == frame_id:
            beg_traces.append(track)
        elif first_frame > frame_id:
            # Take advantage of traces srt wrt (beginning_frame, ending_frame)
            # If the first frame is greater than the current one, we can stop the search

            # track_id = int(track.attrib["id"])
            # print("For frame_id #" + str(frame_id) + ", stopping search at track_id #" + str(track_id))

            break

    return beg_traces


def get_class(annotations, track_id):
    track = annotations.find("./track[@id='" + str(track_id) + "']")
    class_attr = track.attrib["label"]

    return class_attr


def get_present_traces(boxes_els):
    traces = set()

    for box in boxes_els:
        track_id = int(box.attrib["track_id"])
        traces.add(track_id)

    return traces


def get_track_bbox(box_el):
    track_id = int(box_el.attrib["track_id"])

    # x2, y2 = x1 + w, y1 + h
    # i.e. that for the LightTrack notation (xbr, ybr) = (x2, y2), (xtl, ytl) = (x1, y1)
    xbr = round(float(box_el.attrib["xbr"]))
    xtl = round(float(box_el.attrib["xtl"]))
    ybr = round(float(box_el.attrib["ybr"]))
    ytl = round(float(box_el.attrib["ytl"]))

    # print("Instantiating a BBox object...")
    mybb = chpe.BBox(xtl, ytl, xbr, ybr)

    return track_id, mybb


def bbox_str(swig_bbox):
    return "BBox: [ TL(" + str(swig_bbox.xtl) + ", " + str(swig_bbox.ytl) + "), BR(" +str(swig_bbox.xbr) + ", " + str(swig_bbox.ybr) + ") ]"


def to_coord_array(std_human_pose):
    # I want an array of coordinates [x_0, y_0, x_1, y_1, ..., x_N-1, y_N-1]
    p = []

    for i in range(0, len(std_human_pose.keypoints)):
        p.append(std_human_pose.keypoints[i].x)
        p.append(std_human_pose.keypoints[i].y)

    return p


def dump_track(track_id, dict_entry, dump_path):
    coord_arrays = dict_entry["poses"]

    os.makedirs(os.path.join(dump_path, str(track_id)))
    output_x = os.path.join(dump_path, str(track_id), X_DATA)
    output_y = os.path.join(dump_path, str(track_id), Y_DATA)

    with open(output_x, 'w') as f:
        np.savetxt(f, np.array(coord_arrays), fmt='%s', delimiter=',')

    class_id = dict_entry["class"]
    with open(output_y, 'w') as f:
        np.savetxt(f, np.repeat(class_id, len(coord_arrays)), fmt='%s')


def extract_poses_from_videos(proxy_hpe, annotations, video_path,
                              skeletons_path, class_map, debug_prints):

    video_name = os.path.basename(video_path)

    dict = {}
    present_traces = set()
    dumped_traces = set()

    starting_time = datetime.now()
    stop = False
    # The first frame (index 0) is parsed while instantiating the Proxy HPE
    frame_id = 0
    while not stop:
        if debug_prints or DEBUG_FRAME == frame_id+1 or DEBUG_FRAME == frame_id:
            print("------------------------------------------------------------")
            print("FRAME #" + str(frame_id))
            print("------------------------------------------------------------")

        if frame_id % STATUS_FRAME_RATE == 0:
            print("Now processing file " + video_name + ", frame #" + str(frame_id))

        # Determine which traces are present in current frame
        present_boxes = extract_boxes_present_in(annotations, frame_id)
        curr_pres_traces = get_present_traces(present_boxes)

        # Then determine which traces aren't present anymore, wrt to the previous frame
        disappeared_traces = present_traces - curr_pres_traces
        present_traces = curr_pres_traces

        if debug_prints or DEBUG_FRAME == frame_id+1 or DEBUG_FRAME == frame_id:
            print("Disappeared traces: " + str(disappeared_traces))
            print("------------------------------------------------------------")

        # I can dump disappeared traces and free related memory
        # NOTE: do not dump again already dumped traces, if I lost a trace at a certain frame,
        # do not consider it furtherly
        for dis_tr in (disappeared_traces-dumped_traces):
            print("Dumping disappeared track #" + str(dis_tr) + "...")
            dump_track(dis_tr, dict[dis_tr], skeletons_path)
            dumped_traces.add(dis_tr)
            print("Freeing memory of track #" + str(dis_tr) + " data...")
            print("------------------------------------------------------------")
            dict.pop(dis_tr)

        if debug_prints or DEBUG_FRAME == frame_id+1 or DEBUG_FRAME == frame_id:
            print("BOXES PRESENT IN FRAME #" + str(frame_id))
            print("[")

        # Compute pose for each bbox and manage dictionary in memory
        for box in present_boxes:
            if debug_prints or DEBUG_FRAME == frame_id+1 or DEBUG_FRAME == frame_id:
                print_xml(box)

            track_id, bbox = get_track_bbox(box)

            if debug_prints or DEBUG_FRAME == frame_id+1 or DEBUG_FRAME == frame_id:
                print("Handling track_id #" + str(track_id))
                print("Handling bbox: " + bbox_str(bbox))

            if track_id not in dict and track_id not in dumped_traces:
                # New track on frame, I need to insert a new element into the dictionary
                track_class = get_class(annotations, track_id)
                dict[track_id] = {"poses": [], "class": class_map[track_class]}

            # Extract pose from bbox using CV
            if debug_prints or DEBUG_FRAME == frame_id+1 or DEBUG_FRAME == frame_id:
                print("Trying to estimate pose...")
            pose = proxy_hpe.estimate_pose(bbox)

            if debug_prints or DEBUG_FRAME == frame_id+1 or DEBUG_FRAME == frame_id:
                print("Done! Appending pose to dictionary...")
            arr_coord = to_coord_array(pose)
            # if list is not empty
            # NOTE: check if track_id is dict is necessary to avoid error because of lost traces
            if arr_coord and track_id in dict:
                dict[track_id]["poses"].append(arr_coord)

            # stop = proxy.render_bbox(bbox, track_id)

        # Render all poses at the end
        # stop = stop or proxy.render_poses()

        if debug_prints or DEBUG_FRAME == frame_id+1 or DEBUG_FRAME == frame_id:
            print("]")
            print("------------------------------------------------------------")
            print("State of dict: " + str(dict))
            print("------------------------------------------------------------")

        frame_id += 1

        # Read next frame
        stop = stop or (LIMIT_FRAME and frame_id >= N_FRAMES_LIMITATION)
        stop = not proxy_hpe.read() or stop

    # Video running is finished: dump all remaining traces
    print("Pre-processing of " + video_name + " completed. Dumping remaining traces...")
    for track_id in dict.keys():
        print("Dumping track #" + str(track_id) + " poses on input file...")
        print("Dumping track #" + str(track_id) + " class on output file...")
        dump_track(track_id, dict[track_id], skeletons_path)
        dumped_traces.add(dis_tr)

    ending_time = datetime.now()
    print("Starting time: " + str(starting_time))
    print("Ending time: " + str(ending_time))
    print("------------------------------------------------------------")


def sort_traces(cvat_dump, output_file):
    print("Sorting traces of '" + cvat_dump + "'...")

    # Read XML file and extract root element
    tree = ET.parse(cvat_dump)
    root = tree.getroot()

    # Taking all elements of the XML document
    version = root.find("./version")
    meta = root.find("./meta")

    # Create a new XML document with same structure
    new_root = Element("annotations")
    new_root.append(version)
    new_root.append(meta)

    # Get all track elements
    all_traces = root.findall("./track")
    traces = []

    # Remove traces referring to a group class
    for track in root.findall("./track"):
        class_attr = track.attrib["label"]
        # print("Found class_attr: " + class_attr)
        if "together" not in class_attr:
            # print("Adding " + class_attr)
            traces.append(track)

    # Add the track_id attribute to each box sub-element
    for track in traces:
        boxes = track.findall("./box")
        track_id = track.attrib["id"]
        for box in boxes:
            box.set("track_id", track_id)

    # Sort 'track' tags wrt its first frame
    sorted_traces = sorted(traces, key=lambda tr: (get_first_frame(tr), get_last_frame(tr)))

    # Append sorted tracks to new XML document
    for track in sorted_traces:
        new_root.append(track)

    ElementTree(new_root).write(output_file, encoding='utf-8', xml_declaration=True)

    print("Sorted XML written at: " + output_file)
    print("--------------------------------------")


def dump_class_map(class_map, file_path):
    with open(file_path, 'w') as fp:
        json.dump(class_map, fp)


def load_class_map(file_path):
    with open(file_path, 'r') as fp:
        return json.load(fp)


def clean_dataset_folder(skeletons_path):
    trc_paths = [os.path.join(skeletons_path, folder) for folder in os.listdir(skeletons_path)
                 if os.path.isdir(os.path.join(skeletons_path, folder))]

    for folder in trc_paths:
        print("Deleting old trace ", folder, "...")
        shutil.rmtree(folder)
    print("--------------------------------------")


def pre_process(args):

    metadata, labels, class_map, _, _, _ = initialise_variables(class_map_from_tags=True, files_selection=args.files)

    file_path = os.path.join(config['har_dataset_path'], CLASS_MAP_FILE)
    dump_class_map(class_map, file_path)

    # Initialising HPE parameters
    model = './pose_estimation/human_pose_estimation_demo/intel_models/human-pose-estimation-0001/FP32/' \
            'human-pose-estimation-0001.xml'
    target_device = 'CPU'
    show_keypoints_ids = False
    performance_report = False       # This is a flag from the original HPE Demo of OpenVino, not needed
    show_x = False

    i = 1
    print("--------------------------------------")
    print("Video files to be processed: " + str(len(metadata)))
    print("--------------------------------------")
    for md in metadata:
        print("INPUT VIDEO #" + str(i))
        print("Video: " + md.video)
        print("CVAT dump: " + md.cvat_dump)
        print("Tags: " + md.tag)
        print("Skeletons path: " + md.skeletons)
        print("--------------------------------------")

        sort_traces(md.cvat_dump, md.tag)

        clean_dataset_folder(md.skeletons)

        # Open annotations file
        tree = ET.parse(md.tag)
        root = tree.getroot()

        # Open video file with HPE module
        proxy = chpe.CameraHPEProxy(model, target_device, md.video, performance_report, show_keypoints_ids, show_x)

        extract_poses_from_videos(proxy, root, md.video, md.skeletons, class_map, debug_prints=DEBUG_PRINTS)

        i += 1


def fix_list(str_list):
    allowed_decimal_digits = 3

    for i in range(0, len(str_list)):
        point_index = str_list[i].find(".")
        str_list[i] = str_list[i][0:point_index+allowed_decimal_digits+1]
    return str_list


def fix_list_native(str_list):
    str_list = fix_list(str_list)

    float_list = [float(el) for el in str_list]
    return float_list


def unique_values(array):
    return len(np.unique(array))


def remove_multi_class_frames(x, y):
    unique_values_arr = np.apply_along_axis(unique_values, 1, y)
    # print("unique_values_arr", unique_values_arr)
    homog_vectors_index = np.where(unique_values_arr == 1)[0]
    # print("homog_vectors", homog_vectors_index)

    # Reduce y: collapse the lists of same class frames to a single value
    reduced_y = np.apply_along_axis(lambda y: y[0], 1, y[homog_vectors_index])
    return x[homog_vectors_index], reduced_y


def normalise_points(X_dataset):
    x_index = np.arange(0, 36, 2)
    y_index = np.arange(1, 36, 2)

    for batch in X_dataset:
        for pose in batch:
            valid_x = np.where(pose[x_index] != -1.0)
            valid_y = np.where(pose[y_index] != -1.0)

            x_min = pose[x_index][valid_x].min()
            x_max = pose[x_index][valid_x].max()
            y_min = pose[y_index][valid_y].min()
            y_max = pose[y_index][valid_y].max()

            width = x_max - x_min
            height = y_max - y_min

            # Make a copy of x coordinates
            shifted_x = pose[x_index].copy()
            shifted_y = pose[y_index].copy()
            # Shift only valid points
            shifted_x[valid_x] -= x_min
            shifted_y[valid_y] -= y_min

            # Proportionally reduce the pose size if dimensions are bigger
            # than the max allowed
            rf_y = 1
            if height > MAX_HEIGHT:
                rf_y = MAX_HEIGHT / height
            rf_x = 1
            if width > MAX_WIDTH:
                rf_x = MAX_WIDTH / width

            rf = min(rf_x, rf_y)
            if rf < 1:
                shifted_x[valid_x] *= rf
                shifted_y[valid_y] *= rf

            # Replace shifted coordinates in the original array
            pose[x_index] = shifted_x
            pose[y_index] = shifted_y


def compute_new_class_map(class_map, reductions):
    new_class_id = 0
    new_class_map = {}
    id_transl = {}

    # First assign an ID to the reduced classes
    sort_red_keys = [k for k, v in sorted(reductions.items(), key=lambda item: item[1])]
    for red in sort_red_keys:
        if reductions[red] not in new_class_map:
            new_class_map[reductions[red]] = new_class_id
            new_class_id += 1
        id_transl[class_map[red]] = new_class_map[reductions[red]]

    # Then all the others
    class_map_keys = list(class_map.keys())
    class_map_keys.sort()
    for cl in class_map_keys:
        if cl not in sort_red_keys:
            new_class_map[cl] = new_class_id
            id_transl[class_map[cl]] = new_class_id
            new_class_id += 1

    all_keys = [k for k in class_map.keys() if k not in sort_red_keys]
    all_keys.extend(sort_red_keys)


    # print('new_class_map:', new_class_map)
    return new_class_map, id_transl


def reduce_classes(y_dataset, id_transl):
    # Use translation map to change the id of the y dataset
    print("-------- REDUCE CLASSES FUNCTION -------")
    print("id_transl: ", id_transl)
    print("type(y_dataset[0])", type(y_dataset[0]))
    return np.vectorize(id_transl.get)(y_dataset)


# Inspired from https://giov.dev/2018/05/a-window-on-numpy-s-views.html
def native_sliding_window(arr, window_size, step):
    # n_obs = arr.shape[0]
    # pose_len = arr.shape[1]

    n_obs = len(arr)
    pose_len = len(arr[0])

    # validate arguments
    if window_size > n_obs:
        if window_size - n_obs <= 3:
            # print("Less than 3 values missing for a full window: perform padding on last element...")
            padding = [arr[n_obs - 1]]*(window_size - n_obs)
            # print("padding: ", padding)
            arr.extend(padding)
            # print("Arr after appending padding: ", arr)
            return np.array([arr])
        else:
            # raise ValueError(
            #     "Window size must be less than or equal "
            #     "the size of array in first dimension."
            # )
            return np.ones([0, window_size, pose_len])  # empty element: no effect on concatenation
    if step < 0:
        raise ValueError("Step must be positive.")

    n_windows = 1 + int(np.floor((n_obs - window_size) / step))

    windowed_data = []

    for i in range(0, n_windows):
        start_index = i*step
        windowed_data.append(arr[start_index:start_index + window_size])

    return np.array(windowed_data)
    # return np.reshape(windowed_data, newshape=[len(windowed_data), window_size, pose_len])


def extract_slides_from_traces(traces_path, window_length, step, pose_len):
    print("GOING WITH PYTHON NATIVE LISTS FOR PARSING PRE-PROCESSED DATA")
    print("--------------------------------------")
    # step==0 means to obtain non-overlapping contiguous windows
    trc_paths = [os.path.join(traces_path, folder) for folder in os.listdir(traces_path)
                 if os.path.isdir(os.path.join(traces_path, folder))]

    trc_paths.sort()

    traces = np.empty([0, window_length, pose_len])
    labels = np.empty([0])

    for t_path in trc_paths:
        x_file = open(os.path.join(t_path, X_DATA), 'r')
        x_data = [fix_list_native(elem) for elem in [row.split(',') for row in x_file]]
        x_file.close()

        # print("Traces before sliding - x_data.shape: (", len(x_data), ',', len(x_data[0]), ')')
        x_data = native_sliding_window(x_data, window_length, step)
        # print("After sliding - x_data.shape: ", x_data.shape)
        traces = np.concatenate((traces, x_data), axis=0)

        y_file = open(os.path.join(t_path, Y_DATA), 'r')
        label_id = int(y_file.readline())
        y_data = np.repeat(label_id, len(x_data))
        y_file.close()

        # print("No sliding needed for y_data.shape: ", y_data.shape)
        labels = np.concatenate((labels, y_data), axis=0)

    return traces, labels


def extract_labels_for_training(class_map):
    rev_map = {}
    for label in class_map.keys():
        rev_map[class_map[label]] = label
        # print('(Class, ID): (' + str(label) + ', ' + str(class_map[label]) + ')')

    # Watch it! Keys must be ordered by key (value which will be hot encoded)
    ids = list(rev_map.keys())
    ids.sort()
    labels = [rev_map[l] for l in ids]
    return labels, rev_map


def lstm_rnn(_X, _weights, _biases, n_input, n_hidden, restore=False):
    # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

    if not restore:
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        _X = tf.reshape(_X, [-1, n_input])
        # Rectifies Linear Unit activation function used
        _X = tf.nn.relu(tf.matmul(_X, _weights['hidden'], name='relu') + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, n_steps, 0)
        # print("_X: ", _X)

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True, name='lstm_cell_1')
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True, name='lstm_cell_2')
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
        # A single output is produced, in style of "many to one" classifier
        # refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
        lstm_last_output = outputs[-1]
        matmul = tf.matmul(lstm_last_output, _weights['out'], name='my_matmul')
        # save_dict({'my_matmul', matmul.name}, 'matmul')
        # save_dict({'lstm_cells', lstm_cells.name}, 'lstm_cells')
        # save_dict({'states', states.name}, 'states')

    else:
        matmul = tf.get_default_graph().get_tensor_by_name('my_matmul:0')

    # Linear activation
    return matmul + _biases['out']


def extract_batch_size(_train, _labels, _unsampled, batch_size):
    # Fetch a "batch_size" amount of data and labels from "(X|y)_train" data.
    # Elements of each batch are chosen randomly, without replacement, from X_train with corresponding label
    # from Y_train
    # Unsampled_indices keeps track of sampled data ensuring non-replacement.
    # Resets when remaining datapoints < batch_size

    shape = list(_train.shape)
    # print("shape " + str(shape))
    shape[0] = batch_size
    # print("shape " + str(shape))
    batch_s = np.empty(shape)
    batch_labels = np.empty((batch_size))

    for i in range(batch_size):
        # Loop index
        # index = random sample from _unsampled (indices)
        index = random.choice(_unsampled)
        batch_s[i] = _train[index]
        batch_labels[i] = _labels[index]
        _unsampled.remove(index)

    return batch_s, batch_labels, _unsampled


def one_hot(y_, n_classes):
    # One hot encoding of the network outputs
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    # y_ = y_.reshape(len(y_))
    # ATTENZIONE: se manca qualche classe nel set salta tutto
    # n_values = int(np.max(y_)) + 1
    ret_vector = np.eye(n_classes)[np.array(y_, dtype=np.int32)]
    # print("ret_vector.shape " + str(ret_vector.shape))
    return ret_vector  # Returns FLOATS


def initialise_variables(class_map_from_tags=False, files_selection=[],
                         class_map_file=None, reductions=None):

    if not class_map_file:
        class_map_file = os.path.join(config['har_dataset_path'], CLASS_MAP_FILE)

    metadata = initialise_metadata(files_selection)  # If non-empty list or raise an error

    old_class_map = {}
    id_transl = {}
    if class_map_from_tags:
        class_map = extract_class_map(metadata[0].cvat_dump)
    else:
        class_map = load_class_map(file_path=class_map_file)
        # Recompute class map to perform ordering wrt to labels
        old_class_map = class_map
        class_map, id_transl = compute_new_class_map(class_map, reductions)

    labels, id_to_class_map = extract_labels_for_training(class_map)
    return metadata, labels, class_map, old_class_map, id_transl, id_to_class_map


def save_dict(dictionary, path, name):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)


def load_dict(path, name):
    with open(os.path.join(path, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def load_config(config_path):
    global config
    config = {}

    # In increasing order of priority
    # 1. Default configuration
    # 2. Configuration from user defined file

    with open('conf/default_training.json') as def_conf_file:
        config = json.load(def_conf_file)

    if config_path:
        with open(config_path) as file:
            user_config = json.load(file)

    for k in user_config:
        if k in config:
            config[k] = user_config[k]
        else:
            print('Ignoring parameter "', k, '", undefined in "conf/default_training.json"')


def balanced_sample_maker(X, y, uniq_levels, sample_size, random_seed=42):
    if random_seed is not None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        # obs_idx = [idx for idx, val in enumerate(y) if val == level]
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=False).tolist()
        balanced_copy_idx += over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    data_train = X[balanced_copy_idx]
    labels_train = y[balanced_copy_idx]
    if (len(data_train)) == (sample_size*len(uniq_levels)):
        print('number of sampled example ', sample_size*len(uniq_levels), 'number of sample per class ', sample_size,
              ' #classes: ', len(list(set(uniq_levels))))
    else:
        print('number of samples is wrong ')

    labels, values = zip(*Counter(labels_train).items())
    print('number of classes ', len(list(set(labels_train))))
    check = all(x == values[0] for x in values)
    print("check: ", check)
    if check:
        print('Good all classes have the same number of examples')
    else:
        print('Repeat again your sampling your classes are not balanced')
    indexes = np.arange(len(labels))
    # width = 0.5
    # plt.bar(indexes, values, width)
    # plt.xticks(indexes + width * 0.5, labels)
    # plt.show()
    return data_train, labels_train


def mirror_bottlenecks_traces(x_data, y_data):
    uniq_levels = np.unique(y_data)
    traces_per_label_id = {level: sum(y_data == level) for level in uniq_levels}
    min_sample_size = min(traces_per_label_id.values())

    bottlenecks_levels = [l for l in traces_per_label_id if traces_per_label_id[l] == min_sample_size]
    samples_to_flip_index = np.full(list(y_data.shape), False)
    for l in bottlenecks_levels:
        samples_to_flip_index = samples_to_flip_index | (y_data == l)

    # NOTE: if axis=1 is not specified, the sample order is flipped too!!
    inverted_traces = np.flip(x_data[samples_to_flip_index], axis=1)
    x_data = np.concatenate((x_data, inverted_traces), axis=0)
    y_data = np.concatenate((y_data, y_data[samples_to_flip_index]), axis=0)

    return x_data, y_data


def train_rnn(args):
    print("TRAINING PROCEDURE FOR HAR RNN")
    print("--------------------------------------")

    load_config(args.config_path)
    global config
    print('Using the following configuration:')
    print(json.dumps(config, sort_keys=True, indent=4))
    print("--------------------------------------")

    model_path = config['model_output']

    normalise_poses = config['normalise_poses']
    reductions = config['reductions']

    metadata, labels, class_map, old_class_map, id_transl, id_to_class_map = \
        initialise_variables(files_selection=args.files, reductions=reductions)

    print("class_map: ", json.dumps(class_map, indent=2, sort_keys=True))
    print("id_to_class_map: ", json.dumps(id_to_class_map, indent=2, sort_keys=True))
    print("labels: ", labels)

    n_input = n_joints * 2  # Len of OpenPose skeleton array [x_0, y_0, ..., x_17, y_17]
    # -----------------------------------------------------
    # Code from https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input
    # -----------------------------------------------------

    x_slides_data = np.empty([0, n_steps, n_input])
    y_labels_data = np.empty([0])

    print("--------------------------------------")
    print("Video dumps to be processed: ", len(metadata))
    print("--------------------------------------")
    for md in metadata:
        print("Video: " + md.video)
        print("Tag: " + md.tag)
        print("CVAT dump: " + md.cvat_dump)
        print("Skeletons path: " + md.skeletons)
        print("--------------------------------------")

        x_slides_subset, y_labels_subset = extract_slides_from_traces(traces_path=md.skeletons,
                                                                      window_length=n_steps, step=6,
                                                                      pose_len=n_input)

        if normalise_poses:
            normalise_points(x_slides_subset)
            # normalise_points(X_test_subset)

        x_slides_data = np.concatenate((x_slides_data, x_slides_subset), axis=0)

        y_labels_data = np.concatenate((y_labels_data, y_labels_subset), axis=0)

    # print("y_test BEFORE reduction: ", y_labels_data)
    if reductions:
        # Since pre-processing is time-consuming, during pre-processing all classes from CVAT dumps are preserved.
        # In this way, each training session can perform reductions without redoing a pre-processing phase

        y_labels_data = reduce_classes(y_labels_data, id_transl)
        print("old_class_map: ", old_class_map)
        print("id_transl: ", id_transl)
        print("--------------------------------------")

    uniq_levels = np.unique(y_labels_data)
    traces_per_label_id = {level: sum(y_labels_data == level) for level in uniq_levels}
    min_sample_size = min(traces_per_label_id.values())
    uniq_counts = {str(id_to_class_map[label_id]): str(traces_per_label_id[label_id])
                   for label_id in traces_per_label_id}

    print("BEFORE mirroring or balancing the dataset")
    print("--------------------------------------")
    print("Shape of x_data: ", x_slides_data.shape)
    print("Shape of y_data: ", y_labels_data.shape)
    print("uniq_levels: ", uniq_levels)
    print("uniq_counts: ", json.dumps(uniq_counts, indent=2, sort_keys=True))
    print("min_sample_size: ", min_sample_size)
    print("--------------------------------------")

    if config['mirror_few_samples']:
        x_slides_data, y_labels_data = mirror_bottlenecks_traces(x_slides_data, y_labels_data)

        uniq_levels = np.unique(y_labels_data)
        traces_per_label_id = {level: sum(y_labels_data == level) for level in uniq_levels}
        min_sample_size = min(traces_per_label_id.values())
        uniq_counts = {str(id_to_class_map[label_id]): str(traces_per_label_id[label_id])
                       for label_id in traces_per_label_id}

        print("AFTER mirroring the dataset")
        print("--------------------------------------")
        print("Shape of x_data: ", x_slides_data.shape)
        print("Shape of y_data: ", y_labels_data.shape)
        print("uniq_levels: ", uniq_levels)
        print("uniq_counts: ", json.dumps(uniq_counts, indent=2, sort_keys=True))
        print("min_sample_size: ", min_sample_size)
        print("--------------------------------------")

    if config['balance_samples']:
        x_slides_data, y_labels_data = balanced_sample_maker(x_slides_data, y_labels_data, uniq_levels,
                                                             min_sample_size, 33)

        uniq_levels = np.unique(y_labels_data)
        traces_per_label_id = {level: sum(y_labels_data == level) for level in uniq_levels}
        min_sample_size = min(traces_per_label_id.values())
        uniq_counts = {str(id_to_class_map[label_id]): str(traces_per_label_id[label_id])
                       for label_id in traces_per_label_id}

        print("AFTER balancing the dataset")
        print("--------------------------------------")
        print("Shape of x_data: ", x_slides_data.shape)
        print("Shape of y_data: ", y_labels_data.shape)
        print("uniq_levels: ", uniq_levels)
        print("uniq_counts: ", json.dumps(uniq_counts, indent=2, sort_keys=True))
        print("min_sample_size: ", min_sample_size)
        print("--------------------------------------")

    X_train, X_test, y_train, y_test = train_test_split(x_slides_data, y_labels_data, train_size=0.8, random_state=33)

    print("Shape of X_train: ", X_train.shape)
    print("Shape of X_test: ", X_test.shape)
    print("Shape of y_train: ", y_train.shape)
    print("Shape of y_test: ", y_test.shape)
    print("--------------------------------------")

    tf.reset_default_graph()

    # Input Data
    training_data_count = len(X_train)  # 24770 training series (with NO overlap between each serie)
    test_data_count = len(X_test)  # 6193 test series
    n_input = len(X_train[0][0])  # num input parameters per timestep

    print("training_data_count: ", training_data_count)
    print("test_data_count: ", test_data_count)

    n_hidden = 34  # Hidden layer num of features
    n_classes = len(labels)
    # print("labels: ", labels)
    # print("n_classes: ", n_classes)

    # updated for learning-rate decay
    # calculated as: decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    decaying_learning_rate = True
    learning_rate = 0.0025  # used if decaying_learning_rate set to False
    init_learning_rate = 0.005
    decay_rate = 0.96  # the base of the exponential in the decay
    decay_steps = 100000  # used in decay every 60000 steps with a base of 0.96

    global_step = tf.Variable(0, trainable=False, name="global_step")
    lambda_loss_amount = 0.0015

    epochs = config['epochs']
    training_iters = training_data_count * epochs  # Loop 100 times on the dataset, ie 100 epochs
    batch_size = 256
    # display_iter = batch_size*8  # To show test set accuracy during training

    # print("(X shape, y shape, every X's mean, every X's standard deviation)")
    # print(X_train.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    # print("\nThe dataset has not been preprocessed, is not normalised etc")

    # -----------------------------------------------------

    # ------------------------------ BUILD THE NETWORK -------------------------

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')
    # print("n_classes: " + str(n_classes))

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden]), name='weights_hidden'),  # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0), name='weights_out')
    }
    # save_dict({'weights_out': weights['out'].name}, model_path, 'weights_out')
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden]), name='biases_hidden'),
        'out': tf.Variable(tf.random_normal([n_classes]), name='biases_out')
    }

    pred = lstm_rnn(x, weights, biases, n_input, n_hidden)
    # save_dict({'pred', pred.name}, model_path, 'pred_var')

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )  # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
    if decaying_learning_rate:
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size,
                                                   decay_steps, decay_rate, staircase=True)

    # exponentially decayed learning rate
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step,
                                                                             name='adam_optimizer')  # Adam Optimizer

    arg_max_tensor = tf.argmax(pred, axis=1, name='pred_arg_max')
    correct_pred = tf.equal(arg_max_tensor, tf.argmax(y, 1))
    # save_dict({'correct_pred', correct_pred.name}, model_path, 'correct_pred.json')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # ------------------------------ END NETWORK BUILDING ---------------------------------------------

    # Start the TF session
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    if not args.sessiononly:
        # ------------------------------ TRAINING ----------------------------------------------------------
        test_losses = []
        test_accuracies = []
        train_losses = []
        train_accuracies = []

        # Perform Training steps with "batch_size" amount of data at each loop.
        # Elements of each batch are chosen randomly, without replacement, from X_train,
        # restarting when remaining datapoints < batch_size
        step = 1
        time_start = time.time()
        unsampled_indices = list(range(0, len(X_train)))
        # print("Before while: unsampled_indices = " + str(unsampled_indices))

        while step * batch_size <= training_iters:
            # print (sess.run(learning_rate)) #decaying learning rate
            # print (sess.run(global_step)) # global number of iterations
            if len(unsampled_indices) < batch_size:
                unsampled_indices = list(range(0, len(X_train)))
            # print("Before extr_batch: unsampled_indices = " + str(unsampled_indices))
            # _train, _labels, _unsampled, batch_size
            batch_xs, raw_labels, unsampled_indices = extract_batch_size(X_train, y_train, unsampled_indices, batch_size)
            # print("After extr_batch: unsampled_indices = " + str(unsampled_indices))
            batch_ys = one_hot(raw_labels, n_classes)
            # check that encoded output is same length as num_classes, if not, pad it
            if len(batch_ys[0]) < n_classes:
                print("ATTENTION: PADDING SHOULD NOT HAPPEN")
                temp_ys = np.zeros((batch_size, n_classes))
                temp_ys[:batch_ys.shape[0], :batch_ys.shape[1]] = batch_ys
                batch_ys = temp_ys

            # Fit training using batch data
            _, loss, acc = sess.run(
                [optimizer, cost, accuracy],
                feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }
            )
            train_losses.append(loss)
            train_accuracies.append(acc)
            # print("Append element to train_accuracies, now len is: ", len(train_accuracies))

            # Evaluate network only at some steps for faster training:
            # if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
            if ((step*batch_size) % (batch_size*8) == 0) or (step == 1) or (step * batch_size > training_iters):

                # print("step*batch_size % batch_size*8: ", step*batch_size % batch_size*8)
                # To not spam console, show training accuracy/loss in this "if"
                print("Iter #" + str(step*batch_size) +
                      ":  Learning rate = " + "{:.6f}".format(sess.run(learning_rate)) +
                      ":   Batch Loss = " + "{:.6f}".format(loss) +
                      ", Accuracy = {}".format(acc))

                # print("X_test.shape: " + str(X_test.shape))
                # print("y_test.size: " + str(y_test.shape))
                # print("Type of X_test: " + str(type(X_test)))
                # print("Type of one_hot(y_test): " + str(type(one_hot(y_test, n_classes))))

                # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
                loss, acc = sess.run(
                    [cost, accuracy],
                    feed_dict={
                        x: X_test,
                        y: one_hot(y_test, n_classes)
                    }
                )
                test_losses.append(loss)
                test_accuracies.append(acc)
                # print("Append element to test_accuracies, now len is: ", len(test_accuracies))

                print("PERFORMANCE ON TEST SET:             " +
                      "Batch Loss = {}".format(loss) +
                      ", Accuracy = {}".format(acc))

            step += 1

        print("Optimization Finished!")

        # Accuracy for test data
        one_hot_predictions, final_accuracy, final_loss = sess.run(
            [pred, accuracy, cost],
            feed_dict={
                    x: X_test,
                    y: one_hot(y_test, n_classes)
                }
        )

        test_losses.append(final_loss)
        test_accuracies.append(final_accuracy)

        print("FINAL RESULT: " +
              "Loss = {}".format(final_loss) +
              ", Accuracy = {}".format(final_accuracy))
        time_stop = time.time()
        print("TOTAL TIME:  {} seconds".format(time_stop - time_start))
        # ------------------------------ END TRAINING ------------------------------------------------------

        predictions = one_hot_predictions.argmax(1)

        # Results
        print("Testing Accuracy: {}%".format(100 * final_accuracy))

        precision = metrics.precision_score(y_test, predictions, average="weighted")
        recall = metrics.recall_score(y_test, predictions, average="weighted")
        f1_score = metrics.f1_score(y_test, predictions, average="weighted")
        print("Precision: {}%".format(100 * precision))
        print("Recall: {}%".format(100 * recall))
        print("f1_score: {}%".format(100 * f1_score))

        if config['store_model']:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            else:
                for file in [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]:
                    os.remove(os.path.join(model_path, file))

            dump_class_map(class_map, file_path=os.path.join(model_path, CLASS_MAP_FILE))

            with open(os.path.join(model_path, 'model_parameters.json'), 'w') as f:
                json.dump(config, f, indent=2)

            model_statistics = {
                "training_samples": str(training_data_count),
                "test_data_count": str(test_data_count),
                "final_accuracy": str(final_accuracy),
                "final_loss": str(final_loss),
                "training_time_seconds": str(time_stop - time_start),
                "precision": str(precision),
                "recall": str(recall),
                "f1_score": str(f1_score)
            }
            with open(os.path.join(model_path, 'model_statistics.json'), 'w') as f:
                json.dump(model_statistics, f, indent=2)

            plot(batch_size, train_losses, train_accuracies, test_losses, test_accuracies, training_iters,
                 batch_size * 8, predictions, y_test, n_classes, labels, final_accuracy)

            confusion_cross_tabs(y_test, predictions, id_to_class_map)

            # print("--------------------------------------------------")
            # print("TF GLOBAL VARIABLES")
            # print("--------------------------------------------------")
            # for i in tf.global_variables():
            #     print(i)

            print("--------------------------------------------------")
            # print("TF OPERATIONS")
            # print("--------------------------------------------------")
            # for i in sess.graph.get_operations():
            #    print(i)

            session_path = os.path.join(model_path, MODEL_NAME)

            saver = tf.train.Saver()
            saver.save(sess, session_path)
            print("--------------------------------------------------")
            print('Model weights saved at: ', model_path)

    # Close the session
    sess.close()


def dump_plot_args(batch_size, train_losses, train_accuracies, test_losses, test_accuracies,
                   training_iters, display_iter, predictions, y_test, n_classes, labels):

    args = {'batch_size': batch_size,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'training_iters': training_iters,
            'display_iter': display_iter,
            'predictions': predictions,
            'y_test': y_test,
            'n_classes': n_classes,
            'labels': labels}

    path = os.path.join(config['model_output'], '..', 'plot_args')
    try:
        save_dict(args, path, 'args')
    except:
        print('Could not dump plot args')


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.set_label(label=cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")
    # plt.setp(ax.get_xticklabels(), rotation=90)

    # Turn spines off and create white grid.
    # for edge, spine in ax.spines.items():
    #     spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot(batch_size, train_losses, train_accuracies, test_losses, test_accuracies,
         training_iters, display_iter, predictions, y_test, n_classes, labels, accuracy):

    dump_plot_args(batch_size, train_losses, train_accuracies, test_losses, test_accuracies,
                   training_iters, display_iter, predictions, y_test, n_classes, labels)

    plot_dir = os.path.normpath(os.path.join(config['model_output'], '..'))
    model_name = os.path.basename(plot_dir)

    font = {
        'family': 'Bitstream Vera Sans',
        'weight': 'bold',
        'size': 25
    }
    matplotlib.rc('font', **font)

    # Decimal precision of confusion matrices
    float_format = "{x:.0f}"

    # -------------------------- START ACCURACY OVER ITERATIONS ----------------------------

    width = 14
    height = 14
    accuracy_fig = plt.figure(figsize=(width, height))

    indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))/1000
    # plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

    indep_test_axis = np.append(
        np.array(range(batch_size, len(test_losses) * display_iter, display_iter)[:-1]),
        [training_iters]
    )/1000
    # plt.plot(indep_test_axis, np.array(test_losses), "b-", linewidth=2.0, label="Test losses")
    plt.plot(indep_test_axis, np.array(test_accuracies), "b-", linewidth=2.0, label="Test accuracies")
    print("len(test_accuracies): ", len(test_accuracies))
    print("len(train_accuracies): ", len(train_accuracies))

    plt.title("Training session's Accuracy over Iterations")
    plt.legend(loc='lower right', shadow=True)
    plt.ylabel('Training Accuracy')
    plt.xlabel('Training Iteration [k]')

    axes = plt.gca()
    axes.set_ylim([0.0, 1.0])
    # plt.show()d

    plt.savefig(os.path.join(plot_dir, model_name + '_acc.png'), dpi=300, bbox_inches='tight')
    plt.close(accuracy_fig)
    # -------------------------- END ACCURACY OVER ITERATIONS ----------------------------

    # ---------------------- START CONFUSION MATRIX ON TOTAL -----------------------------
    print("")
    print("Confusion Matrix:")
    print("Created using test set of {} datapoints, normalised to % of each class in the test dataset".format(
        len(y_test)))
    confusion_matrix = metrics.confusion_matrix(y_test, predictions).astype(np.float)

    # np.set_printoptions(precision=2)

    # print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100

    conf_fig, ax = plt.subplots(figsize=(16, 16))

    im, cbar = heatmap(normalised_confusion_matrix, labels, labels,
                       ax=ax, cmap=plt.cm.rainbow, cbarlabel="Predictions over total [%]")

    texts = annotate_heatmap(im, valfmt=float_format)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.set_title("Confusion matrix \n(Normalised over number of total test data)")
    im.set_clim(0, MAX_Y_CONF_MATRIX)
    conf_fig.tight_layout()

    plt.savefig(os.path.join(plot_dir, model_name + '_cm_tot.png'), dpi=300, bbox_inches='tight')
    plt.close(conf_fig)

    # -------------------------- END CONFUSION MATRIX ON TOTAL ---------------------------
    
    # -------------------------- START CONFUSION MATRIX ON ROW ----------------------------

    # Confusion matrix normalised on row (true label)
    denom = confusion_matrix.sum(axis=1, keepdims=True)
    norm_on_row_cm = np.divide(confusion_matrix, denom, out=np.zeros_like(confusion_matrix), where=denom != 0) * 100
    conf_fig, ax = plt.subplots(figsize=(16, 16))

    im, cbar = heatmap(norm_on_row_cm, labels, labels,
                       ax=ax, cmap=plt.cm.rainbow, cbarlabel="Samples of true labels [%]")

    texts = annotate_heatmap(im, valfmt=float_format)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.set_title("Confusion matrix \n(Normalised over samples per class)")
    im.set_clim(0, 100)
    conf_fig.tight_layout()

    plt.savefig(os.path.join(plot_dir, model_name + '_cm_row.png'), dpi=300, bbox_inches='tight')
    plt.close(conf_fig)
    
    # -------------------------- END CONFUSION MATRIX ON ROW ----------------------------

    # ------------------------- START CONFUSION MATRIX ON COL ---------------------------

    denom = confusion_matrix.sum(axis=0)
    # Confusion matrix normalised on row (true label)
    norm_on_col_cm = np.divide(confusion_matrix, denom, out=np.zeros_like(confusion_matrix), where=denom != 0) * 100
    conf_fig, ax = plt.subplots(figsize=(16, 16))

    im, cbar = heatmap(norm_on_col_cm, labels, labels,
                       ax=ax, cmap=plt.cm.rainbow, cbarlabel="Samples of predicted labels [%]")

    texts = annotate_heatmap(im, valfmt=float_format)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.set_title("Confusion matrix \n(Normalised over predictions per class)")
    im.set_clim(0, 100)
    conf_fig.tight_layout()

    plt.savefig(os.path.join(plot_dir, model_name + '_cm_col.png'), dpi=300, bbox_inches='tight')
    plt.close(conf_fig)

    # -------------------------- END CONFUSION MATRIX ON COL ----------------------------


def confusion_cross_tabs(y_test, y_pred, id_to_class_map):
    global config
    # create some data
    y_true_pd = pd.Series([id_to_class_map[_id] for _id in y_test])
    y_pred_pd = pd.Series([id_to_class_map[_id] for _id in y_pred])

    pd.crosstab(y_true_pd, y_pred_pd, rownames=['True'], colnames=['Predicted'], margins=True)\
        .to_csv(os.path.join(config['model_output'], 'pd_absolute_confusion_matrix.csv'))

    pd.crosstab(y_true_pd, y_pred_pd, rownames=['True'], colnames=['Predicted'])\
        .apply(lambda r: 100.0 * r / r.sum())\
        .to_csv(os.path.join(config['model_output'], 'pd_relative_confusion_matrix.csv'))


'''
def restore_from_file(args):
    global config
    print("DEBUG FUNCTION - RESTORING SESSION FROM FILE")
    print("--------------------------------------")

    # NOTE: I am reading 'model_output', because it is the attribute of a previous run training session
    # so I reference the output specified in that config
    model_path = config['model_output']

    with open(os.path.join(model_path, 'model_parameters'), 'r') as model_conf:
        config_params = json.load(model_conf)

    # Restore the trained model
    # delete the current graph
    tf.reset_default_graph()

    # import the graph from the file
    imported_graph = tf.train.import_meta_graph(args.model_input)

    # list all the tensors in the graph
    for tensor in tf.get_default_graph().get_operations():
        print(tensor.name)

    # run the session
    with tf.Session() as sess:
        # restore the saved vairable
        imported_graph.restore(sess, get_default_tf_session_path())
        print("Model restored from checkpoint.")

        if args.files:
            # Continue to train with selected files
            metadata, labels, class_map = \
                initialise_variables(class_map_from_tags=False,
                                     files_selection=args.files,
                                     class_map_file=os.path.join(model_path, CLASS_MAP_FILE))

            id_transl = config_params['id_transl']

            # save_dict({'labels': labels}, model_path, 'labels')

            n_input = 36  # Len of OpenPose skeleton array [x_0, y_0, ..., x_17, y_17]
            # -----------------------------------------------------
            # Code from https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input
            # -----------------------------------------------------

            x_slides_data = np.empty([0, n_steps, n_input])

            y_labels_data = np.empty([0, n_steps, 1])

            for md in metadata:
                print("Video: " + md.video)
                print("Tag: " + md.tag)
                print("CVAT dump: " + md.cvat_dump)
                print("Skeletons path: " + md.skeletons)
                print("--------------------------------------")

                x_slides_subset, y_labels_subset = extract_slides_from_traces(traces_path=md.skeletons,
                                                                              window_length=n_steps,
                                                                              step=6, pose_len=n_input)
                if config_params["normalise_poses"]:
                    normalise_points(x_slides_subset)

                x_slides_data = np.concatenate((x_slides_data, x_slides_subset), axis=0)
                # X_test = np.concatenate((X_test, X_test_subset), axis=0)

                y_labels_data = np.concatenate((y_labels_data, y_labels_subset), axis=0)

            print("id_transl: ", id_transl)

            # TODO: here apply reduction from configuration
            if config_params["reductions"]:
                y_labels_data = reduce_classes(y_labels_data, id_transl)
                # y_test = reduce_classes(y_test, id_transl)

            X_train, X_test, y_train, y_test = train_test_split(x_slides_data, y_labels_data, train_size=0.8,
                                                                random_state=33)

            # print("Shape of X_test: " + str(X_test.shape))
            # print("Shape of y_test: " + str(y_test.shape))
            # print("X_test: ", X_test)
            # print("y_test: ", y_test)
            # print("y_train: ", y_train)

            # Remove frames series where there is more than one class
            # X_train, y_train = remove_multi_class_frames(X_train, y_train)
            # X_test, y_test = remove_multi_class_frames(X_test, y_test)

            # Input Data
            training_data_count = len(X_train)
            test_data_count = len(X_test)
            n_input = len(X_train[0][0])

            n_hidden = 34  # Hidden layer num of features
            n_classes = len(labels)
            # print("labels: ", labels)
            # print(n_classes)

            # updated for learning-rate decay
            # calculated as: decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
            decaying_learning_rate = True
            learning_rate = 0.0025  # used if decaying_learning_rate set to False
            init_learning_rate = 0.005
            decay_rate = 0.96  # the base of the exponential in the decay
            decay_steps = 100000  # used in decay every 60000 steps with a base of 0.96

            global_step = tf.get_default_graph().get_tensor_by_name('global_step:0')
            lambda_loss_amount = 0.0015

            epochs = config_params['epochs']
            training_iters = training_data_count * epochs  # Loop 100 times on the dataset, ie 100 epochs
            batch_size = 256
            # display_iter = batch_size*8  # To show test set accuracy during training

            # print("(X shape, y shape, every X's mean, every X's standard deviation)")
            # print(X_train.shape, y_test.shape, np.mean(X_test), np.std(X_test))
            # print("\nThe dataset has not been preprocessed, is not normalised etc")

            # -----------------------------------------------------

            # ------------------------------ BUILD THE NETWORK -------------------------
            # x, y, pred, optimizer, accuracy, cost, learning_rate = build_network(n_steps, n_input, n_classes, n_hidden,
            #                                                                     decaying_learning_rate, learning_rate,
            #                                                                     init_learning_rate, global_step, batch_size,
            #                                                                     decay_steps, decay_rate, lambda_loss_amount)

            # Graph input/output
            x = tf.get_default_graph().get_tensor_by_name('x:0')
            y = tf.get_default_graph().get_tensor_by_name('y:0')
            # print("n_classes: " + str(n_classes))

            # Graph weights
            weights = {
                'hidden': tf.get_default_graph().get_tensor_by_name(name='weights_hidden:0'),  # Hidden layer weights
                'out': tf.get_default_graph().get_tensor_by_name(name='weights_out:0')
            }
            biases = {
                'hidden': tf.get_default_graph().get_tensor_by_name(name='biases_hidden:0'),
                'out': tf.get_default_graph().get_tensor_by_name(name='biases_out:0')
            }

            pred = lstm_rnn(x, weights, biases, n_input, n_hidden, restore=True)

            # Loss, optimizer and evaluation
            l2 = lambda_loss_amount * sum(
                tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
            )  # L2 loss prevents this overkill neural network to overfit the data
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2  # Softmax loss
            if decaying_learning_rate:
                learning_rate = tf.train.exponential_decay(init_learning_rate, global_step * batch_size, decay_steps,
                                                           decay_rate, staircase=True)

            # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps) #exponentially decayed learning rate
            optimizer = tf.get_default_graph().get_tensor_by_name(name='adam_optimizer:0')  # Adam Optimizer

            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # ------------------------------ END NETWORK BUILDING ---------------------------------------------

            # ------------------------------ TRAINING ----------------------------------------------------------
            # train_the_network(sess, X_train, y_train, X_test, y_test, batch_size, training_iters, n_classes,
            #                   learning_rate, x, y, pred, optimizer, accuracy, cost, normalise_poses, simple_walk)
            test_losses = []
            test_accuracies = []
            train_losses = []
            train_accuracies = []

            # Perform Training steps with "batch_size" amount of data at each loop.
            # Elements of each batch are chosen randomly, without replacement, from X_train,
            # restarting when remaining datapoints < batch_size
            step = 1
            time_start = time.time()
            unsampled_indices = list(range(0, len(X_train)))
            # print("Before while: unsampled_indices = " + str(unsampled_indices))

            while step * batch_size <= training_iters:
                # print (sess.run(learning_rate)) #decaying learning rate
                # print (sess.run(global_step)) # global number of iterations
                if len(unsampled_indices) < batch_size:
                    unsampled_indices = list(range(0, len(X_train)))
                # print("Before extr_batch: unsampled_indices = " + str(unsampled_indices))
                # _train, _labels, _unsampled, batch_size
                batch_xs, raw_labels, unsampled_indices = extract_batch_size(X_train, y_train, unsampled_indices,
                                                                             batch_size)
                # print("After extr_batch: unsampled_indices = " + str(unsampled_indices))
                batch_ys = one_hot(raw_labels, n_classes)
                # check that encoded output is same length as num_classes, if not, pad it
                if len(batch_ys[0]) < n_classes:
                    print("ATTENTION: PADDING SHOULD NOT HAPPEN")
                    temp_ys = np.zeros((batch_size, n_classes))
                    temp_ys[:batch_ys.shape[0], :batch_ys.shape[1]] = batch_ys
                    batch_ys = temp_ys

                # print('batch_xs.shape: ', batch_xs.shape)

                # Fit training using batch data
                _, loss, acc = sess.run(
                    [optimizer, cost, accuracy],
                    feed_dict={
                        x: batch_xs,
                        y: batch_ys
                    }
                )
                train_losses.append(loss)
                train_accuracies.append(acc)

                # Evaluate network only at some steps for faster training:
                # if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
                if ((step * batch_size) % (batch_size * 8) == 0) or (step == 1) or (step * batch_size > training_iters):
                    # To not spam console, show training accuracy/loss in this "if"
                    print("Iter #" + str(step * batch_size) +
                          ":  Learning rate = " + "{:.6f}".format(sess.run(learning_rate)) +
                          ":   Batch Loss = " + "{:.6f}".format(loss) +
                          ", Accuracy = {}".format(acc))

                    # print("X_test.shape: " + str(X_test.shape))
                    # print("y_test.size: " + str(y_test.shape))
                    # print("Type of X_test: " + str(type(X_test)))
                    # print("Type of one_hot(y_test): " + str(type(one_hot(y_test, n_classes))))

                    # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
                    loss, acc = sess.run(
                        [cost, accuracy],
                        feed_dict={
                            x: X_test,
                            y: one_hot(y_test, n_classes)
                        }
                    )
                    test_losses.append(loss)
                    test_accuracies.append(acc)
                    print("PERFORMANCE ON TEST SET:             " +
                          "Batch Loss = {}".format(loss) +
                          ", Accuracy = {}".format(acc))

                step += 1

            print("Optimization Finished!")

            # Accuracy for test data
            one_hot_predictions, accuracy, final_loss = sess.run(
                [pred, accuracy, cost],
                feed_dict={
                    x: X_test,
                    y: one_hot(y_test, n_classes)
                }
            )

            test_losses.append(final_loss)
            test_accuracies.append(accuracy)

            print("Normalisation: ", config_params['normalise_poses'])
            print("Simple walk: ", config_params['simple_walk'])
            print("FINAL RESULT: " +
                  "Loss = {}".format(final_loss) +
                  ", Accuracy = {}".format(accuracy))
            time_stop = time.time()
            print("TOTAL TIME:  {} seconds".format(time_stop - time_start))
            # ------------------------------ END TRAINING ------------------------------------------------------
'''


def multiple_training(args):
    for conf_path in args.conf_path_list:
        parser = argparse.ArgumentParser()

        parser.add_argument('-conf', dest='config_path', type=str)

        parser.add_argument('-files', type=str, nargs='*',
                            help='List of file names (in CVAT dumps path) to train.')
        parser.add_argument('-sessiononly', action='store_true', required=False,
                            help='DEBUG OPT: do not train, setup the TF session only')

        args_string = '-conf ' + conf_path
        sub_args = parser.parse_args(args_string.split())
        print("sub_args: ", sub_args)
        train_rnn(sub_args)


def load_data(metadata, normalise_poses, reductions, class_map, old_class_map, id_transl, id_to_class_map,
              n_input=n_joints*2):
    x_slides_data = np.empty([0, n_steps, n_input])
    y_labels_data = np.empty([0])

    for md in metadata:
        print("Video: " + md.video)
        print("Tag: " + md.tag)
        print("CVAT dump: " + md.cvat_dump)
        print("Skeletons path: " + md.skeletons)
        print("--------------------------------------")

        x_slides_subset, y_labels_subset = extract_slides_from_traces(traces_path=md.skeletons,
                                                                      window_length=n_steps, step=6,
                                                                      pose_len=n_input)

        if normalise_poses:
            normalise_points(x_slides_subset)
            # normalise_points(X_test_subset)

        x_slides_data = np.concatenate((x_slides_data, x_slides_subset), axis=0)

        y_labels_data = np.concatenate((y_labels_data, y_labels_subset), axis=0)

    # print("y_test BEFORE reduction: ", y_labels_data)
    if reductions:
        # Since pre-processing is time-consuming, during pre-processing all classes from CVAT dumps are preserved.
        # In this way, each training session can perform reductions without redoing a pre-processing phase

        y_labels_data = reduce_classes(y_labels_data, id_transl)
        print("old_class_map: ", old_class_map)
        print("id_transl: ", id_transl)
        print("class_map: ", class_map)
        print("--------------------------------------")

    uniq_levels = np.unique(y_labels_data)
    traces_per_label_id = {level: sum(y_labels_data == level) for level in uniq_levels}
    min_sample_size = min(traces_per_label_id.values())
    uniq_counts = {str(id_to_class_map[label_id]): str(traces_per_label_id[label_id])
                   for label_id in traces_per_label_id}

    print("BEFORE mirroring or balancing the dataset")
    print("--------------------------------------")
    print("Shape of x_data: ", x_slides_data.shape)
    print("Shape of y_data: ", y_labels_data.shape)
    print("uniq_levels: ", uniq_levels)
    print("uniq_counts: ", json.dumps(uniq_counts, indent=2, sort_keys=True))
    print("min_sample_size: ", min_sample_size)
    print("--------------------------------------")

    if config['mirror_few_samples']:
        x_slides_data, y_labels_data = mirror_bottlenecks_traces(x_slides_data, y_labels_data)

        uniq_levels = np.unique(y_labels_data)
        traces_per_label_id = {level: sum(y_labels_data == level) for level in uniq_levels}
        min_sample_size = min(traces_per_label_id.values())
        uniq_counts = {str(id_to_class_map[label_id]): str(traces_per_label_id[label_id])
                       for label_id in traces_per_label_id}

        print("AFTER mirroring the dataset")
        print("--------------------------------------")
        print("Shape of x_data: ", x_slides_data.shape)
        print("Shape of y_data: ", y_labels_data.shape)
        print("uniq_levels: ", uniq_levels)
        print("uniq_counts: ", json.dumps(uniq_counts, indent=2, sort_keys=True))
        print("min_sample_size: ", min_sample_size)
        print("--------------------------------------")

    if config['balance_samples']:
        x_slides_data, y_labels_data = balanced_sample_maker(x_slides_data, y_labels_data, uniq_levels,
                                                             min_sample_size, 33)

        uniq_levels = np.unique(y_labels_data)
        traces_per_label_id = {level: sum(y_labels_data == level) for level in uniq_levels}
        min_sample_size = min(traces_per_label_id.values())
        uniq_counts = {str(id_to_class_map[label_id]): str(traces_per_label_id[label_id])
                       for label_id in traces_per_label_id}

        print("AFTER balancing the dataset")
        print("--------------------------------------")
        print("Shape of x_data: ", x_slides_data.shape)
        print("Shape of y_data: ", y_labels_data.shape)
        print("uniq_levels: ", uniq_levels)
        print("uniq_counts: ", json.dumps(uniq_counts, indent=2, sort_keys=True))
        print("min_sample_size: ", min_sample_size)
        print("--------------------------------------")

    return x_slides_data, y_labels_data


def plot_statistics(args):
    print("Plotting utility of the pre-processed data set")
    print("--------------------------------------")

    for conf_path in args.conf_path_list:
        load_config(conf_path)
        global config
        normalise_poses = config['normalise_poses']
        reductions = config['reductions']

        metadata, labels, class_map, old_class_map, id_transl, id_to_class_map = initialise_variables(reductions=reductions)

        x_data, y_data = load_data(metadata, normalise_poses, reductions, class_map, old_class_map, id_transl,
                                   id_to_class_map)

        stats = []
        for class_id in id_to_class_map:
            # For each class
            class_subset = x_data[y_data == class_id]
            # Extract the overall number of missing data points
            np.count_nonzero(class_subset == -1.0) // 2

            # Number of missing data points per pose, shape (n_traces, n_poses)
            miss_pts_per_pose = np.count_nonzero(class_subset == -1.0, axis=2) // 2
            miss_pts_per_trace = miss_pts_per_pose.sum(axis=1)

            incomplete_traces = np.count_nonzero(miss_pts_per_trace)
            perc_inc_traces = round(incomplete_traces / len(class_subset)*100, 2)
            incomplete_poses = np.count_nonzero(miss_pts_per_pose)
            perc_inc_poses = round(incomplete_poses / (len(class_subset) * class_subset.shape[1])*100, 2)

            # Average number of missing points for incomplete traces
            avg_missing_joints = miss_pts_per_trace[miss_pts_per_trace.nonzero()].mean()
            perc_avg_missing_joints = round(avg_missing_joints/(n_steps*n_joints)*100, 2)

            # [Class, Incomplete traces, Ratio inc. tr., Incomp. poses, Ratio inc. p., Avg. missing joints
            stats.append([id_to_class_map[class_id], incomplete_traces, perc_inc_traces,
                          incomplete_poses, perc_inc_poses, avg_missing_joints, perc_avg_missing_joints])

        table = PrettyTable(['Class', 'Inc. Traces', 'Perc. Inc. Traces', 'Inc. Poses', 'Perc. Inc. Poses',
                             'Avg. Miss. Jts per Inc. Tr.', 'Perc. Miss. Jts. per Inc. Tr.'])
        stats.sort()
        for row in stats:
            table.add_row(row)

        print(table)
        print("--------------------------------------")

        table_txt = table.get_string()
        with open(os.path.join(config['model_output'], 'missing_joints_stats.txt'), 'w') as file:
            file.write(table_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''HPE-to-HAR for robots - RNN training utility''')

    subpars = parser.add_subparsers(help='Commands to pre-process or train the RNN')

    parser_stats = subpars.add_parser('stat', help='Plot statistics about the pre-processed data set.')
    parser_stats.add_argument('-conf', dest='conf_path_list', type=str, nargs='*', required=True,
                              help='List of path to configuration files for training.')
    parser_stats.set_defaults(func=plot_statistics)

    # Create the parser for the "pre-process" command
    parser_preproc = subpars.add_parser('pre-process',
                                        help='Pre-process data: prepare CVAT dumps and then extract poses '
                                             'using OpenPose-based HPE module.')

    parser_preproc.add_argument('-files', type=str, nargs='*',
                                help='List of file names (in CVAT dumps '
                                     'path) to pre-process.')
    # parser_preproc.add_argument('-i', dest='dump_path', type=str, required=False,
    #                             default=CVAT_DUMPS_PATH,
    #                             help='Path containing the original CVAT dumps.')
    # parser_preproc.add_argument('-o', dest='pose_path', type=str, required=False,
    #                             help='Path where the pose dataset will be stored.')

    parser_preproc.set_defaults(func=pre_process)

    # Create the parser for the "train" command
    parser_train = subpars.add_parser('train', help='Train the RNN using pre-processed data.')

    parser_train.add_argument('-conf', dest='config_path', type=str, required=False,
                              help='Path to configuration file for training.')
    # parser_train.add_argument('-normalise', action='store_true', required=False,
    #                          help='Normalise skeletal poses into square BBoxes.')
    # parser_train.add_argument('-simplewalk', action='store_true', required=False,
    #                          help='No difference between slow/fast walking.')
    # parser_train.add_argument('-store', action='store_true', required=False,
    #                           help='Store the model weights after training.')
    # parser_train.add_argument('-epochs', type=int, default=100, required=False,
    #                           help='Number of epochs.')
    parser_train.add_argument('-files', type=str, nargs='*',
                              help='List of file names (in CVAT dumps '
                                   'path) to train.')
    parser_train.add_argument('-sessiononly', action='store_true', required=False,
                              help='DEBUG OPT: do not train, setup the TF session only')
    # parser_train.add_argument('-o', dest='model_output', type=str, required=False,
    #                           help='Path where the TF session will be stored.')

    parser_train.set_defaults(func=train_rnn)

    parser_mult = subpars.add_parser('multiple-training', help='Launch multiple training sessions of the HAR RNN'
                                                               'using different configurations.')

    parser_mult.add_argument('-conf', dest='conf_path_list', type=str, nargs='*', required=True,
                             help='List of path to configuration files for training.')

    parser_mult.set_defaults(func=multiple_training)

    '''
    parser_restore = subpars.add_parser('restore', help='Restore RNN weights from checkpoint.')
    parser_restore.add_argument('-conf', dest='config_path', type=str, required=False,
                              help='Path to configuration file for training.')
    # parser_restore.add_argument('-i', dest='model_input', type=str, required=False,
    #                             help='Path containing the TF session.')
    # parser_restore.add_argument('-files', type=str, nargs='*',
    #                             help='List of file names (in CVAT dumps '
    #                                  'path) to continue the training.')
    parser_restore.set_defaults(func=restore_from_file)
    '''

    args = parser.parse_args()
    args.func(args)
