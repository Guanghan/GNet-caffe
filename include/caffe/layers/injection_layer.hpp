#ifndef CAFFE_INJECTION_LAYER_HPP_
#define CAFFE_INJECTION_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Data layer: transform ground truth data into nutrition.
 *
 * This layer computes insightful hand-crafted features inferred from gt,
 * and inject into the network as auxiallary nutrition for the network.
 *
 * [1] Guanghan Ning, "Human Pose Estimation." arxiv preprint
 *     arXiv:1502.03167 (2017).
 */

template <typename Dtype>
class InjectionLayer : public CPMDataLayer<Dtype> {
 public:
  explicit InjectionLayer(const LayerParameter& param)
      : CPMDataLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Injection"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_INJECTION_LAYER_HPP_
