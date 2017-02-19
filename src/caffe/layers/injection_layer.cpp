#include <algorithm>
#include <vector>

#include "caffe/layers/injection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Create an Injection Class


INSTANTIATE_CLASS(InjectionLayer);
REGISTER_LAYER_CLASS(Injection);

}  // namespace caffe
