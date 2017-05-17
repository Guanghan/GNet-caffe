'''
    Test edge features for Human Pose Estimation
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Feb, 2016
'''

#include "HOG.h"
#include "utils.h"

bool test_getLimbFromJoints(Point2f joint_1, Point2f joint_2);
bool test_getLineParams(Point2f joint_1, Point2f joint_2);

bool test_blockNormalization(Point2f joint_1, Point2f joint_2);
bool test_orientBinningForCells(Point2f joint_1, Point2f joint_2);
