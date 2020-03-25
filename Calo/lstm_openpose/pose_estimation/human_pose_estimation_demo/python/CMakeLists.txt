# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# This is a CMake example for Python

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS highgui QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# outdir for .py file
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})
# outdir for .so file
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

SET(CMAKE_SWIG_FLAGS "")

set(SOURCES ../src/camera_hpe_proxy.cpp
            ../src/camera_hpe.cpp
            ../src/human_pose.cpp
            ../src/human_pose_estimator.cpp
            ../src/peak.cpp
            ../src/render_human_pose.cpp
            ../src/bbox.cpp
            ../src/render_bbox.cpp)

SET_SOURCE_FILES_PROPERTIES(chpe_proxy.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(chpe_proxy.i PROPERTIES SWIG_FLAGS "-py3;-extranative")

# -includeall option forces SWIG to look for headers only in the specified
# include directories -> the 'standard' include dirs are excluded!
#SET_SOURCE_FILES_PROPERTIES(chpe_proxy.i PROPERTIES SWIG_FLAGS "-includeall")

SWIG_ADD_MODULE(chpe python chpe_proxy.i ${SOURCES})
SWIG_LINK_LIBRARIES(chpe ${PYTHON_LIBRARIES} ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} ${LIB_FOLDER} gflags)

if(UNIX)
  SWIG_LINK_LIBRARIES(chpe ${LIB_DL} pthread)
endif()
