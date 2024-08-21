#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <algorithm>
using std::max;

namespace py = pybind11;

void minimize_prefill_iteration_time(
    py::ssize_t num_new_req,
    py::ssize_t num_instances,
    py::array_t<double> min_iteration_times,
    py::array_t<int32_t> min_iteration_times_index,
    py::array_t<int64_t> req_input_prefix_sum,
    py::array_t<int64_t> req_input_square_prefix_sum,
    py::array_t<double> inverted_req_input_prefix_sum,
    py::array_t<int64_t> left_token_prefix_sum,
    py::array_t<double> predictor_parameters) {
    
    auto min_iteration_times_ = min_iteration_times.mutable_unchecked<2>();
    auto min_iteration_times_index_ = min_iteration_times_index.mutable_unchecked<3>();

    auto req_input_prefix_sum_ = req_input_prefix_sum.unchecked<1>();
    auto req_input_square_prefix_sum_ = req_input_square_prefix_sum.unchecked<1>();
    auto inverted_req_input_prefix_sum_ = inverted_req_input_prefix_sum.unchecked<1>();
    auto left_token_prefix_sum_ = left_token_prefix_sum.unchecked<1>();

    auto predictor_parameters_ = predictor_parameters.unchecked<2>();

    min_iteration_times_(0, 0) = 0;
    for (py::ssize_t num_served_req = 1; num_served_req <= num_new_req; ++num_served_req) {
        for (py::ssize_t num_used_instances = 1; num_used_instances <= num_instances; ++num_used_instances) {
            
            const int32_t prev_last_batch_size = min_iteration_times_index_(num_served_req, num_used_instances - 1, 0);
            const py::ssize_t max_last_batch_size = (prev_last_batch_size != -1)
                ? prev_last_batch_size : num_served_req;
            
            for (py::ssize_t last_batch_size = 1; last_batch_size <= max_last_batch_size; ++last_batch_size) {

                const int32_t prev_last_used_instances = min_iteration_times_index_(num_served_req - 1, num_used_instances, 1);
                const py::ssize_t min_last_used_instances = (prev_last_used_instances != -1)
                    ? prev_last_used_instances : 1;
                
                for (py::ssize_t last_used_instances = num_used_instances; last_used_instances >= min_last_used_instances; --last_used_instances) {

                    const int32_t num_prev_served_req = num_served_req - last_batch_size;
                    const int64_t req_input_sum = req_input_prefix_sum_(num_served_req - 1)
                        - ((num_prev_served_req > 0)
                            ? req_input_prefix_sum_(num_prev_served_req - 1): 0);

                    
                    const int32_t num_prev_used_instances = num_used_instances - last_used_instances;
                    const int64_t left_token_sum = left_token_prefix_sum_(num_used_instances - 1)
                        - ((num_prev_used_instances > 0)
                            ? left_token_prefix_sum_(num_prev_used_instances - 1): 0);
                    
                    if (req_input_sum > left_token_sum) {
                        break;
                    }

                    const int64_t req_input_square_sum = req_input_square_prefix_sum_(num_served_req - 1)
                        - ((num_prev_served_req > 0)
                            ? req_input_square_prefix_sum_(num_prev_served_req - 1): 0);
                    const double A = predictor_parameters_(last_used_instances, 0);
                    const double B = predictor_parameters_(last_used_instances, 1);
                    const double C = predictor_parameters_(last_used_instances, 2);
                    const double cur_batch_time = A + B * req_input_sum + C * req_input_square_sum;

                    const double inverted_req_input_sum = inverted_req_input_prefix_sum_(num_served_req - 1)
                        - ((num_prev_served_req > 0)
                            ? inverted_req_input_prefix_sum_(num_prev_served_req - 1): 0);
                    const double total_iteration_time = cur_batch_time * inverted_req_input_sum + min_iteration_times_(num_prev_served_req, num_prev_used_instances);

                    if (total_iteration_time < min_iteration_times_(num_served_req, num_used_instances)) {
                        min_iteration_times_(num_served_req, num_used_instances) = total_iteration_time;
                        min_iteration_times_index_(num_served_req, num_used_instances, 0) = last_batch_size;
                        min_iteration_times_index_(num_served_req, num_used_instances, 1) = last_used_instances;
                    }

                }
            }
        }
    }
}

PYBIND11_MODULE(longserve_c_scheduler, m) {
    m.doc() = "Some C++ implementations for LongServe scheduler";

    m.def("minimize_prefill_iteration_time",
        &minimize_prefill_iteration_time,
        py::arg("num_new_req").noconvert(),
        py::arg("num_instances").noconvert(),
        py::arg("min_iteration_times").noconvert(),
        py::arg("min_iteration_times_index").noconvert(),
        py::arg("req_input_prefix_sum").noconvert(),
        py::arg("req_input_square_prefix_sum").noconvert(),
        py::arg("inverted_req_input_prefix_sum").noconvert(),
        py::arg("left_token_prefix_sum").noconvert(),
        py::arg("predictor_parameters").noconvert(),
        "Minimize prefill iteration time."
    );
}