# hpe-to-har

### Requirements

- Ubuntu 16.04 LTS
- [Intel OpenVINO Toolkit](https://software.intel.com/en-us/openvino-toolkit)
  - Installation instructions [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#configure-model-optimizer)
- CMake
- [SWIG](http://www.swig.org)
- python >= 3.5
- matplotlib
- numpy
- pandas
- tensorflow >= 1, <2

### Setup a running Anaconda virtual environment for RNN HAR training

```sh
conda create env -n cat_har
conda install -n cat_har "python>=3.5" 
conda install -c anaconda -n cat_har numpy pandas
conda install -c conda-forge -n cat_har matplotlib tensorflow
conda activate cat_har
```

### Data set pre-processing

The pre-processing can de divided in two phases, the first one is performed offline and the second one "online" before a training session.

Offline phase, for each tagged video:
1. Ordering of corresponding CVAT tagged traces with regard to (starting_frame, ending_frame) to allow the subsequent
   extraction of skeletons in a single video pass with Lightweight OpenPose HPE.
2. Lightweight OpenPose run on individual frames containing multiple bounding boxes on the raw video: the extraction of skeleton has been done by maintaining in memory queues for each trace tagged into the CVAT dump. The output is a folder for each trace, containing the time sequence of skeletons, each of 18 joints position (x, y) relative to the bounding box origin, and the class ID representing the tagged action being performed.

This phase can be launched using the `pre-process` sub-command of the `training.py` CLI:

```
python training.py pre-process -h

usage: training.py pre-process [-h] [-files [FILES [FILES ...]]]

optional arguments:
  -h, --help            show this help message and exit
  -files [FILES [FILES ...]]
                        List of file names (in CVAT dumps path) to pre-
                        process.
```

To cope with the highly varying length of the average FPT, during the online pre-processing phase of a training session, sliding windows of fixed length have been extracted from each trace.

Online phase, for each video folder, containing the traces of that video:
1. Loading in memory of time sequences of skeletons produced by the offline phase using python native data structures
2. Fixed lenght sliding windows extraction (lenght=32 frames, offset=6 frames => overlapping=81.25%)  and data set structuring as a Numpy array

This phase can be configured with a JSON file under 'conf/' and launched with the `train` sub-command of the `training.py` CLI:

```
python training.py train -h

usage: training.py train [-h] [-conf CONFIG_PATH] [-files [FILES [FILES ...]]]
                         [-sessiononly]

optional arguments:
  -h, --help            show this help message and exit
  -conf CONFIG_PATH     Path to configuration file for training.
  -files [FILES [FILES ...]]
                        List of file names (in CVAT dumps path) to train.
```

### Run real-time HAR on camera using OpenVINO HPE

1. Create a build directory
```sh
mkdir pose_estimation/build
```

2. Go to the created build directory
```sh
cd pose_estimation/build
```
3. Run CMake
```sh
cmake -DCMAKE_BUILD_TYPE=Release ../
```
4. Run make
```sh
make
```

5. Go back to project directory
```sh
cd ../..
```

6. Run python script
```sh
python3 run.py
```
