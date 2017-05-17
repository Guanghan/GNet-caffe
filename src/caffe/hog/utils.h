'''
    Extract edge features for Human Pose Estimation
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Feb, 2016
'''

#ifndef UTILS_H
#define UTILS_H
//#define USE_OPENCV

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <utility>

using namespace cv;

struct LineParams {
  double A, B, C, D;
};

struct Limb {
  Point2f joint_begin, joint_end;
  float x_begin, x_end, y_begin, y_end;
  float orientTan;
  float wid, len;
  LineParams line_params;
};

void getLimbFromJoints(Point2f joint_1, Point2f joint_2, Limb& region);
void getLineParams(Point2f joint_begin, Point2f joint_end, LineParams& line_params);

void blockNormalization(float* histogram, int hist_length);
float* orientBinningForCells(Mat &img, Limb region, int num_of_cells, Mat map_horizontal, Mat map_vertical);
bool checkPixelInLimb(Point pixel, LineParams line_params, float range);
void binningPixel(Point pixel, Mat map_horizontal, Mat map_vertical,
                                                   float* histogram);
int checkWhichBin(float response_horizontal, float response_vertical);
void dispGradientMaps(Mat map_horizontal, Mat map_vertical, bool wait_key = true);
void genGradientMaps(Mat img,
                     Mat gradient_map_horizontal,
                     Mat gradient_map_vertical);
void filter_with_kernel(Mat img_gray,
                        Mat kernel,
                        Mat filtered);
bool assert_maps_size(Mat map_1, Mat map_2);

#endif  // USE_OPENCV
#endif //UTILS_H
