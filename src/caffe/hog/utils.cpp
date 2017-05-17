'''
    Extract edge features for Human Pose Estimation
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Feb, 2016
'''

#include "math.h"

void getLimbFromJoints(Point2f joint_1, Point2f joint_2, Limb& region){
  region.joint_begin = (joint_1.y > joint_2.y)?joint_2:joint_1;
  region.joint_end = (joint_1.y > joint_2.y)?joint_1:joint_2;

  region.orientTan = (joint_1.y - joint_2.y) / (joint_1.x - joint_2.x);
  region.len = cv::norm(joint_1 - joint_2);
  region.wid = 10.0;

  region.x_begin = (joint_1.x > joint_2.x)?joint_2.x:joint_1.x;
  region.x_end = (joint_1.x > joint_2.x)?joint_1.x:joint_2.x;
  region.y_begin = (joint_1.y > joint_2.y)?joint_2.y:joint_1.y;
  region.y_end = (joint_1.y > joint_2.y)?joint_1.y:joint_2.y;
  region.x_begin -= region.wid;
  region.x_end += region.wid;
  region.y_begin -= region.wid;
  region.y_end += region.wid;

  getLineParams(region.joint_begin, region.joint_end, region.line_params); //@
}


void getLineParams(Point2f joint_begin, Point2f joint_end, LineParams& line_params){
  int y1 = joint_begin.y;
  int y2 = joint_end.y;
  int x1 = joint_begin.x;
  int x2 = joint_end.x;

  line_params.A = y2 - y1;
  line_params.B = x1 - x2;
  line_params.C = (-1)* x1 * line_params.A - y1 * line_params.B;
  line_params.D = sqrt(line_params.A*line_params.A + line_params.B*line_params.B);
}

//------------------------------------------------------------------------------
/* 3. Descriptor for a block (a pair of joints) + Block Normalization */
void blockNormalization(float* histogram, int hist_length){
  float total = 0.00000001;

  for (int i = 0; i < hist_length; i++){
    total += (histogram[i] * histogram[i]);
  }
  total = sqrt(total);

  if (total == 0) return;

  for (int i = 0; i < hist_length; i++){
    histogram[i] /= total;
  }
}

//------------------------------------------------------------------------------
/* 2. Orientation Binning*/
float* orientBinningForCells(Mat &image, Limb region, int num_of_cells, Mat map_horizontal, Mat map_vertical){
  // Allocate space for histogram
  const int num_of_bins = 8;
  int hist_lenth_per_cell = num_of_bins;
  int hist_length = hist_lenth_per_cell * num_of_cells;

  float* histogram = new float[hist_length];
  for (int i = 0; i < hist_length; i++){
    histogram[i] = 0;
  }

  LineParams line_params = region.line_params;
  float range = region.wid;

  int y_begin = region.y_begin > 0 ? region.y_begin : 0;
  int y_end = region.y_end < map_vertical.rows ? region.y_end : map_vertical.rows;
  int x_begin = region.x_begin > 0 ? region.x_begin : 0;
  int x_end = region.x_end < map_vertical.cols ? region.x_end : map_vertical.cols;

  for (int y = y_begin; y < y_end; y++)
    for (int x = x_begin; x < x_end; x++){
      circle(image, Point(x, y), 1.0, Scalar(125, 0, 255), -1, 8 );

      if (checkPixelInLimb(Point(x, y), line_params, range)) {
        binningPixel(Point(x, y), map_horizontal, map_vertical, histogram);
        circle(image, Point(x, y), 1.0, Scalar(255, 255, 255), -1, 8 );
      }
    }

  // namedWindow("limb region", CV_WINDOW_AUTOSIZE);
  // imshow("limb region", image);
  // waitKey(0);

  return histogram;
}


bool checkPixelInLimb(Point pixel, LineParams line_params, float range){
  // if the dist between pixel and the line (traverse joints with orientTan) within wid
  int x0 = pixel.x;
  int y0 = pixel.y;

  double dist = abs(line_params.A * x0 + line_params.B * y0 + line_params.C) / line_params.D;

  if (dist <= range){
    return true;
  }
  else{
    return false;
  }
}


void binningPixel(Point pixel, Mat map_horizontal, Mat map_vertical,
                                                   float* histogram){
  float response_horizontal = map_horizontal.at<float>(pixel.y, pixel.x);
  float response_vertical = map_vertical.at<float>(pixel.y, pixel.x);

  int bin = checkWhichBin(response_horizontal, response_vertical);

  histogram[bin] += 1; //magnitude;
}


int checkWhichBin_old(float response_horizontal, float response_vertical){
  float orient = atan2(response_vertical, response_horizontal);
  float orient_in_degrees = orient * 180 / 3.14;
  //std::cout<<"orient_in_degrees"<<orient_in_degrees<<std::endl;
  std::cout<<"orient: "<<orient<<std::endl;
  std::cout<<"response_vertical: "<< response_vertical << std::endl;
  std::cout<<"response_horizontal: "<< response_horizontal << std::endl;

  float step = 180.0 / 8;
  int bin = 0;
  for (float bar_high = step; bar_high < 180; bar_high += step){
    float bar_low = bar_high - step;
    if ((abs(orient_in_degrees) >= bar_low) && (abs(orient_in_degrees) < bar_high)){
      return bin;
    }
    bin ++;
  }

  return -1;
}

int checkWhichBin(float response_horizontal, float response_vertical){
  if (response_vertical < 0 || response_horizontal< 0){
    //std::cout<< "grad_horizontal: " << response_horizontal << ", grad_vertical: "<< response_vertical << std::endl;
  }
  //std::cout<< "grad_horizontal: " << response_horizontal << ", grad_vertical: "<< response_vertical << std::endl;

  float orientTan = abs(1.0 * response_vertical / response_horizontal);
  int bin = -1;

  if (orientTan >= 1.0) {
    if (response_vertical >=0 && response_horizontal >= 0){
      bin = 1;
    }
    else if (response_vertical >= 0 && response_horizontal < 0){
      bin = 2;
    }
    else if (response_vertical < 0 && response_horizontal >= 0){
      bin = 6;
    }
    else{
      bin = 5;
    }
  }
  else{
    if (response_vertical >=0 && response_horizontal >= 0){
      bin = 0;
    }
    else if (response_vertical >= 0 && response_horizontal < 0){
      bin = 3;
    }
    else if (response_vertical < 0 && response_horizontal >= 0){
      bin = 7;
    }
    else{
      bin = 4;
    }
  }
  return bin;
}

//------------------------------------------------------------------------------
/* 1. Gradient Maps */
void dispGradientMaps(Mat map_horizontal, Mat map_vertical, bool wait_key){
  const char* windowName_h = "gradient map horizontal";
  const char* windowName_v = "gradient map vertical";

  namedWindow(windowName_h, CV_WINDOW_AUTOSIZE);
  imshow(windowName_h, map_horizontal);

  namedWindow(windowName_v, CV_WINDOW_AUTOSIZE);
  imshow(windowName_v, map_vertical);

  if (wait_key == true)  waitKey(0);
  else waitKey(200);
}


void genGradientMaps(Mat img,
                     Mat gradient_map_horizontal, Mat gradient_map_vertical){
  // Check whether maps are of uniform resolution
  assert_maps_size(img, gradient_map_horizontal);
  assert_maps_size(img, gradient_map_vertical);

  Mat kernel_horizontal = (Mat_<double>(1, 3) << -1, 0, 1);
  Mat kernel_vertical = (Mat_<double>(3, 1) << 1, 0, -1);

  filter_with_kernel(img, kernel_horizontal, gradient_map_horizontal);
  filter_with_kernel(img, kernel_vertical, gradient_map_vertical);
}


void filter_with_kernel(Mat img_gray, Mat kernel,
                                      Mat filtered){
  // this is just an interface for opencv's filter2D function
  assert_maps_size(img_gray, filtered);

  filter2D(img_gray,
           filtered,
           CV_16S, //depth
           kernel,
           Point(-1, -1), //at the middle
           0);  //delta
}


bool assert_maps_size(Mat map_1, Mat map_2){
  const int rows_1 = map_1.rows;
  const int cols_1 = map_1.cols;

  const int rows_2 = map_2.rows;
  const int cols_2 = map_2.cols;

  assert((rows_1 == rows_2) && (cols_1 == cols_2));
}
