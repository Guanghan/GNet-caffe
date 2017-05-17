'''
    Test edge features for Human Pose Estimation
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Feb, 2016
'''

#include <iostream>
#include <stdlib.h>
#include "test_HOG.h"

bool test_getLimbFromJoints(Point2f joint_1, Point2f joint_2){
  Limb region;
  bool success = false;

  getLimbFromJoints(joint_1, joint_2, region);

  if (&region != NULL){
    cout<<"\t joint_begin: ("<<region.joint_begin.x<<", "<<region.joint_begin.y<<")"<<endl;
    cout<<"\t joint_end: ("<<region.joint_end.x<<", "<<region.joint_end.y<<")"<<endl;

    cout<<"\t orientTan: "<<region.orientTan<<endl;
    cout<<"\t len: "<<region.len<<endl;
    cout<<"\t wid: "<<region.wid<<endl;

    cout<<"\t x_begin: "<<region.x_begin<<endl;
    cout<<"\t x_end: "<<region.x_end<<endl;
    cout<<"\t y_begin: "<<region.y_begin<<endl;
    cout<<"\t y_end: "<<region.y_end<<endl;

    success = true;
  }
  return success;
}


bool test_getLineParams(Point2f joint_1, Point2f joint_2){
  LineParams line_params;
  bool success = false;

  getLineParams(joint_1, joint_2, line_params);

  if(&line_params != NULL){
    cout<<"\t line_params.A: "<< line_params.A<<endl;
    cout<<"\t line_params.B: "<< line_params.B<<endl;
    cout<<"\t line_params.C: "<< line_params.C<<endl;
    cout<<"\t line_params.D: "<< line_params.D<<endl;
    success = true;
  }
  return success;
}


bool test_dispGradientMaps(){
  const char* img_path = "imgs/im1031.jpg";
  Mat img_gray = imread(img_path, 0);

  namedWindow("org img", CV_WINDOW_AUTOSIZE);
  imshow("org img", img_gray);

  Mat gradient_map_vertical = img_gray.clone();
  Mat gradient_map_horizontal = img_gray.clone();
  genGradientMaps(img_gray, gradient_map_horizontal, gradient_map_vertical);
  dispGradientMaps(gradient_map_horizontal, gradient_map_vertical, true);

  return true;
}


bool test_blockNormalization(Point2f joint_1, Point2f joint_2){
  const char* img_path = "imgs/soccer.jpg";
  Mat img_gray = imread(img_path, 0);

  Limb region;
  getLimbFromJoints(joint_1, joint_2, region);

  namedWindow("org img", CV_WINDOW_AUTOSIZE);
  imshow("org img", img_gray);

  Mat gradient_map_vertical = img_gray.clone();
  Mat gradient_map_horizontal = img_gray.clone();
  genGradientMaps(img_gray, gradient_map_horizontal, gradient_map_vertical);
  dispGradientMaps(gradient_map_horizontal, gradient_map_vertical, false);

  float* histogram = orientBinningForCells(img_gray, region, 1, gradient_map_horizontal, gradient_map_vertical);

  blockNormalization(histogram, 8);

  bool success = false;
  if(histogram != NULL){
    for (int i = 0; i< 8; i++){
      cout<<"\t histogram["<< i << "] = " << histogram[i] << endl;
      if (~isnan(histogram[i]))  success = true;
    }
  }
  return success;
}


bool test_orientBinningForCells(Point2f joint_1, Point2f joint_2){
  const char* img_path = "imgs/soccer.jpg";
  Mat img_gray = imread(img_path, 0);

  namedWindow("org img", CV_WINDOW_AUTOSIZE);
  imshow("org img", img_gray);

  Mat gradient_map_vertical = img_gray.clone();
  Mat gradient_map_horizontal = img_gray.clone();
  genGradientMaps(img_gray, gradient_map_horizontal, gradient_map_vertical);
  dispGradientMaps(gradient_map_horizontal, gradient_map_vertical, true);

  Limb region;

  getLimbFromJoints(joint_1, joint_2, region);
  float* histogram = orientBinningForCells(img_gray, region, 1, gradient_map_horizontal, gradient_map_vertical);
  waitKey(0);

  bool success = false;
  if(histogram != NULL){
    for (int i = 0; i< 8; i++){
    cout<<"\t histogram["<< i << "] = " << histogram[i] << endl;
    }
    success = true;
  }
  return success;
}
