#include <algorithm>
#include <vector>

#include "caffe/layers/injection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Create an Injection Class


INSTANTIATE_LAYER_GPU_FUNCS(InjectionLayer);

}  // namespace caffe
