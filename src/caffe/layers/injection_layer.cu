#include <algorithm>
#include <vector>

#include "caffe/layers/injection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Create an Injection Class

/* Just like cpm_data_layer, the cuda version is not provided */


INSTANTIATE_LAYER_GPU_FUNCS(InjectDataLayer);

}  // namespace caffe
