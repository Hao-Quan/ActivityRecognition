// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include <vector>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>

#include "human_pose_estimation_demo.hpp"
#include "std_human_pose.hpp"
#include "camera_hpe_proxy.hpp"


using namespace InferenceEngine;
using namespace human_pose_estimation;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    std::cout << "[ INFO ] Parsing input parameters" << std::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return EXIT_SUCCESS;
        }

        CameraHPEProxy chpe_proxy(FLAGS_m, FLAGS_d, FLAGS_i, FLAGS_pc, FLAGS_k, FLAGS_x);


        bool stop=false;
        long int frame = 0;
        do {

            std::vector<StdHumanPose> poses = chpe_proxy.estimate_poses();

            if (FLAGS_r) {
                std::stringstream outputstream;
                outputstream << "---------------------------" << std::endl;
                outputstream << "FRAME #" << frame << std::endl;
                outputstream << "---------------------------" << std::endl;
                for (size_t i=0; i<poses.size(); i++) {
                    StdHumanPose const& pose = poses[i];
                    outputstream << "Skeleton #" << i+1 << std::endl;
                    outputstream << std::fixed << std::setprecision(0);
                    for (size_t i = 0; i < pose.keypoints.size(); i++) {
                        Point2D const& point = pose.keypoints[i];
                        outputstream << point.x << "," << point.y << ",";
                        outputstream << std::setprecision(2) << point.score << " ";
                    }
                    outputstream << "| " << pose.score << std::endl;
                    outputstream << "---------------------------" << std::endl;
                }
                std::cout << outputstream.str() << std::endl;
            }

            if (FLAGS_no_show) {
                continue;
            }

            stop = chpe_proxy.render_poses();
            frame++;

        } while (chpe_proxy.read() && !stop);
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[ INFO ] Execution successful" << std::endl;
    return EXIT_SUCCESS;
}
