import argparse
import collections
import json
import numpy as np
import os
import pose_estimation.build.human_pose_estimation_demo.python.chpe as chpe
import tensorflow as tf
import time

from functools import partial
from training import lstm_rnn, MAX_HEIGHT, MAX_WIDTH, \
                     MODEL_DIR, MODEL_NAME, MODEL_FILE_NAME, \
                     load_class_map, CLASS_MAP_FILE

# -------------------------------
# -------- CODE START -----------
# -------------------------------

keypoints_number = 18
TIME_SERIES_LEN = 32
FORGET_THRESHOLD = 5


# os.environ["CUDA_VISIBLE_DEVICES"]="0"


def load_network(model_path):
    print("Restore from file")

    with open(os.path.join(model_path, 'model_parameters.json'), 'r') as model_conf:
        config_params = json.load(model_conf)

    label_to_index_map = load_class_map(os.path.join(model_path, CLASS_MAP_FILE))

    # Build reverse map
    index_to_label_map = {}
    for k in label_to_index_map:
        index_to_label_map[label_to_index_map[k]] = k

    # Restore the trained model
    # delete the current graph
    tf.reset_default_graph()

    # import the graph from the file
    imported_graph = tf.train.import_meta_graph(os.path.join(model_path, MODEL_FILE_NAME))

    # list all the tensors in the graph
    # for tensor in tf.get_default_graph().get_operations():
    #     print(tensor.name)

    return imported_graph, config_params, label_to_index_map, index_to_label_map


def invalid_frame_id():
    return lambda: -1


def extract_single_input_for_rnn(kpts_seq):
    single_batch = np.empty([1, TIME_SERIES_LEN, keypoints_number*2])
    single_batch[0] = kpts_seq
    # Shape is now [1, 32, 36]
    return single_batch


def extract_batch_input_for_rnn(tracks_dict, full_tracks):
    global keypoints_number
    shape = [len(tracks_dict.keys()), TIME_SERIES_LEN, keypoints_number*2]
    batch_xs = np.empty(shape)
    for i in range(0, len(full_tracks)):
        full_tr = full_tracks[i]
        batch_xs[i] = tracks_dict[full_tr]

    return batch_xs


def normalise_points(keypoints, remove_score=False):
    # keypoints = [x_0, y_0, sc_0, ..., x_17, y_17, sc_17]
    global keypoints_number
    step = 3
    pose = np.array(keypoints)
    if remove_score:
        step = 2
        my_ind = np.array([i % 3 != 2 for i in range(0, keypoints_number * 3)])
        pose = pose[my_ind]

    keypoints_len = keypoints_number * step

    x_index = np.arange(0, keypoints_len, step)
    y_index = np.arange(1, keypoints_len, step)

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

    return pose


def update_tracks_dict(tracks_dict, track_id, keypoints):
    # First normalise the keypoints wrt the standard bbox [MAX_WIDTH, MAX_HEIGHT]
    # and remove the pointwise score
    pose = normalise_points(keypoints, remove_score=True)

    # Then append the pose to its track series
    tracks_dict[track_id].append(pose)


def run_on_camera(chpe_proxy, rnn_weights_path, stat_path):
    st_time_total = time.time()

    imported_graph, config_params, label_to_index_map, index_to_label_map = load_network(rnn_weights_path)

    # Dictionary of tracks, where the key is track_id and the value a list of keypoint lists
    tracks_dict = collections.defaultdict(partial(collections.deque, maxlen=TIME_SERIES_LEN))

    # run the session
    with tf.Session() as sess:
        # Restore session from the saved graph
        # NOTE: only name of the model needed, without '.meta' extension
        imported_graph.restore(sess, os.path.join(rnn_weights_path, MODEL_NAME))
        print("Model restored from checkpoint.")

        # Graph input/output
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        # y = tf.get_default_graph().get_tensor_by_name('y:0')

        biases = {
            'hidden': tf.get_default_graph().get_tensor_by_name(name='biases_hidden:0'),
            'out': tf.get_default_graph().get_tensor_by_name(name='biases_out:0')
        }

        pred_tensor = lstm_rnn(x, None, biases, None, None, restore=True)
        arg_max_tensor = tf.get_default_graph().get_tensor_by_name(name='pred_arg_max:0')

        # Buffer of detections of the previous frame
        bbox_dets_list_q = collections.deque(maxlen=2)

        img_id = 0
        next_id = 0
        stop = False
        statistics = []
        while not stop:
            end_to_end_start_time = time.time()

            ''' At each frame: (1) Call the HPE module to extract poses;
                               (2) Tracking via Spatial Consistency
                               (3) Inference on LSTM to classify the performed activity in traces '''

            ''' (1): Call the HPE module and extract poses in the current frame '''
            hpe_start_time = time.time()
            poses = chpe_proxy.estimate_poses()
            hpe_time = time.time() - hpe_start_time

            bbox_dets_list = []  # BBoxes extracted in the current frame

            # Dictionary of classes, where the key is track_id and the value the index of the class label
            track_to_det_id_map = collections.defaultdict(int)

            human_candidates = extract_bboxes_from_hpe(poses)
            num_dets = len(human_candidates)

            if img_id >= 1:  # First frame (with img_id==0) does not have a previous frame
                bbox_prev_frame_list = bbox_dets_list_q[-1]
            else:
                bbox_prev_frame_list = []

            tracking_start_time = time.time()
            for det_id in range(num_dets):
                # Obtain bbox position and track id
                bbox_det_x1y1x2y2 = human_candidates[det_id]

                # Enlarge bbox by 20% with same center position
                scaled_bbox_x1y1x2y2 = enlarge_bbox(bbox_det_x1y1x2y2, scale=0.2)
                keypoints = stdpose_kpts_to_coco_list(poses[det_id])

                if img_id == 0:  # First frame, all ids are assigned automatically
                    track_id = next_id
                    next_id += 1
                else:
                    ''' (2) Tracking via Spatial Consistency '''
                    track_id, match_index = track_via_spatial_consist(scaled_bbox_x1y1x2y2,
                                                                      bbox_prev_frame_list)

                    if track_id != -1:  # Match found
                        # if candidate from prev frame matched, prevent it from matching another
                        del bbox_prev_frame_list[match_index]

                        # This has been tracked, so I add track_id on the proxy poses before rendering,
                        # This way each skeleton will have a specific color depending on its track_id
                        track_to_det_id_map[track_id] = det_id
                        chpe_proxy.set_id(det_id, track_id)
                    else:
                        # Matching not found, assign a new ID
                        track_id = next_id
                        next_id += 1

                # Append keypoints to the time series of keypoints
                update_tracks_dict(tracks_dict, track_id, keypoints)

                # update current frame bbox
                bbox_det_dict = {
                    "track_id": track_id,
                    "bbox": scaled_bbox_x1y1x2y2,
                    "forget_index": 0
                 }
                bbox_dets_list.append(bbox_det_dict)

            # At this point in bbox_prev_frame_list there are only not matched traces
            # Buffer all entries with forget_index less than forget tolerance
            entries_to_buff = [bbox_entry for bbox_entry in bbox_prev_frame_list
                               if bbox_entry["forget_index"] < FORGET_THRESHOLD]

            # Forget all traces that have not found a match for more than FORGET_THRESHOLD frames
            forget_tracks = [bbox_entry["track_id"] for bbox_entry in bbox_prev_frame_list
                             if (bbox_entry["forget_index"] >= FORGET_THRESHOLD)]

            # Increment forget_index
            for bbox_entry in entries_to_buff:
                bbox_entry["forget_index"] += 1
                bbox_dets_list.append(bbox_entry)

            # Delete 'old' tracks, i.e. not seen since FORGET_THRESHOLD frames
            for k in forget_tracks:
                print('Deleting track #', k)
                del tracks_dict[k]

            tracking_time = time.time() - tracking_start_time

            ''' (3) HAR via inference on LSTM for full traces '''

            # Here performs inference real-time, X_live is the series of frames for each track_id
            full_traces = [tr for tr in tracks_dict if len(tracks_dict[tr]) == TIME_SERIES_LEN
                           and tr in track_to_det_id_map]

            har_time = 0
            for full_tr in full_traces:
                # Extract full traces from dictionary and feed them to the LSTM network
                x_live = extract_single_input_for_rnn(tracks_dict[full_tr])

                # Get the action predicted
                har_start_time = time.time()
                [arg_max_float], pred = sess.run([arg_max_tensor, pred_tensor], feed_dict={x: x_live})
                har_time += (time.time() - har_start_time)
                class_index = int(arg_max_float)

                # Get bbox of the person on which we performed the activity prediction:
                det_id_of_track = track_to_det_id_map[full_tr]
                bbox_det_dict = bbox_dets_list[det_id_of_track]
                assert (bbox_det_dict["track_id"] == full_tr)

                # Rendering bounding box and activity prediction
                chpe_bbox = to_chpe_bbox(bbox_det_dict["bbox"])
                chpe_proxy.render_bbox(chpe_bbox, class_index, index_to_label_map[class_index])

            # Append current detections to buffer
            bbox_dets_list_q.append(bbox_dets_list)

            # Updating stats, for each frame:
            # [Detections, HPE Time, Tracking Time, HAR Time, N. Full Traces, End-to-end Time]
            end_to_end_time = time.time() - end_to_end_start_time
            statistics.append([num_dets, hpe_time, tracking_time, har_time, len(full_traces), end_to_end_time])

            img_id += 1
            stop = chpe_proxy.render_poses()

            # Read next frame and repeat
            stop = not chpe_proxy.read() or stop

        # Get total running time
        total_run_time = time.time() - st_time_total

        if not os.path.exists(stat_path):
            os.makedirs(stat_path)

        npstats = np.array(statistics)
        np.save(os.path.join(stat_path, 'np_stats'), npstats)

        model_statistics = {
            "analysed_frames": str(img_id),
            "total_run_time": str(total_run_time)
        }
        # Dump runtime statistics to file
        with open(os.path.join(stat_path, 'other.json'), 'w') as f:
            json.dump(model_statistics, f, indent=2)


def print_poses(poses):
    n = 1
    print('-----------------------------------')
    for pose in poses:
        print("Skeleton #" + str(n))
        print('-----------------------------------')
        for i in range(0, len(pose.keypoints)):
            print('{:.0f}'.format(pose.keypoints[i].x) + "," +
                  '{:.0f}'.format(pose.keypoints[i].y) + "," +
                  '{:.2f}'.format(pose.keypoints[i].score))
        n += 1
        print('-----------------------------------')


# Taken from https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
# TODO: check if it is needed
def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def track_via_spatial_consist(bbox_cur_frame, bbox_list_prev_frame):
    thresh = 0.3
    max_iou_score = 0
    max_index = -1

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        iou_score = get_iou_score(bbox_cur_frame, bbox_prev_frame)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = bbox_index

    if max_iou_score > thresh:
        track_id = bbox_list_prev_frame[max_index]["track_id"]
        return track_id, max_index
    else:
        return -1, -1


def get_iou_score(bbox_a, bbox_b):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou_score = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou_score


def enlarge_bbox(bbox, scale):
    assert (scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0:
        margin_x = 2
    if margin_y < 0:
        margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x = 0
        max_x = 2
        min_y = 0
        max_y = 2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def stdpose_to_np_xy(pose):
    # Create two numpy arrays of x and y coordinates
    # replacing invalid point coordinate -1 with NaN
    x = np.zeros(len(pose.keypoints))
    y = np.zeros(len(pose.keypoints))
    s = np.zeros(len(pose.keypoints))
    for i in range(len(pose.keypoints)):
        x[i] = pose.keypoints[i].x
        y[i] = pose.keypoints[i].y
        s[i] = pose.keypoints[i].score
    x[x == -1.] = np.nan
    y[y == -1.] = np.nan
    return x, y, s


def stdpose_kpts_to_coco_list(pose):
    keypoints = keypoints_number * 3 * [0]
    for i in range(0, len(pose.keypoints)):
        keypoints[i * 3] = pose.keypoints[i].x
        keypoints[i * 3 + 1] = pose.keypoints[i].y
        keypoints[i * 3 + 2] = pose.keypoints[i].score
    return keypoints


def from_lighttrack_to_coord_list(keypoints):
    global keypoints_number

    coord_list = np.array(keypoints)

    # Exclude each element whose position is a multiple of 3
    # i.e. [ True, True, False, True, True, False, ... ]
    my_ind = np.array([i % 3 != 2 for i in range(0, keypoints_number * 3)])
    return coord_list[my_ind]


def from_stdpose_to_coord_list(pose):
    # Produce a list of coordinates [x_0, y_0, x_1, y_1, ..., x_N-1, y_N-1]
    global keypoints_number
    coord_list = [0] * 2 * keypoints_number
    for i in range(0, len(pose.keypoints)):
        coord_list[i * 2 + 0] = pose.keypoints[i].x
        coord_list[i * 2 + 1] = pose.keypoints[i].y

    return coord_list


def compute_bbox_x1y1x2y2(x, y):
    x[x == -1.] = np.nan
    y[y == -1.] = np.nan
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    return [xmin, ymin, xmax, ymax]


# def compute_bbox_xywh(x, y):
#    bbox_x1y1x2y2 = compute_bbox_x1y1x2y2(x, y)
#    return x1y1x2y2_to_xywh(bbox_x1y1x2y2)


def extract_bboxes_from_hpe(poses):
    # Calculate the bounding boxes from the array of poses
    human_candidates = []
    for pose in poses:
        x, y, _ = stdpose_to_np_xy(pose)
        bbox = compute_bbox_x1y1x2y2(x, y)
        human_candidates.append(bbox)
    return human_candidates


def to_chpe_bbox(py_bbox_x1y1x2y2):
    xtl, ytl, xbr, ybr = py_bbox_x1y1x2y2
    int_xtl = int(round(xtl))
    int_ytl = int(round(ytl))
    int_xbr = int(round(xbr))
    int_ybr = int(round(ybr))
    chpe_bbox = chpe.BBox(int_xtl, int_ytl, int_xbr, int_ybr)
    return chpe_bbox


def bbox_invalid(bbox):
    if bbox == [0, 0, 2, 2]:
        return True
    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[2] > 2000 or bbox[3] > 2000:
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Real-time lightweight human pose estimation python demo with
                       tracking feature.
                       This is a Python wrapper of the HPE C++ implementation
                       provided into the Intel OpenVINO toolkit, with modified
                       Lighttrack light tracker.''')
    parser.add_argument('--m', type=str, default='./pose_estimation/human_pose_estimation_demo/intel_models/'
                                                 'human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml',
                        help='Optional. Path to the Human Pose Estimation model (.xml) file.')
    parser.add_argument('--i', type=str, default='cam',
                        help='Optional. Path to a video. Default value is "cam" to work with camera.')
    parser.add_argument('--k', action='store_true', help='Optional. Show keypoints index on the rendered image.')
    parser.add_argument('--x', action='store_true', help='Optional. Show x coordinates on the rendered image.')
    parser.add_argument('--w', type=str, required=False, help='Path containing the TF session files.',
                        default='training_sessions/redux_vs_norm/redux_norm/model')
    parser.add_argument('--o', dest='stat_path', type=str, help='Output path of statistics.',
                        default='./runtime_stats')

    args = parser.parse_args()

    model = args.m
    target_device = 'CPU'
    video_path = args.i
    show_keypoints_ids = args.k
    performance_report = False  # This is a flag from the original HPE Demo of OpenVino, not needed
    show_x = args.x
    weights_path = args.w

    proxy = chpe.CameraHPEProxy(model, target_device, video_path, performance_report, show_keypoints_ids, show_x)
    run_on_camera(proxy, weights_path, args.stat_path)
