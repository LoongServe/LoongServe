#include <cassert>
#include <cinttypes>
#include <cstdio>

#include <nccl.h>
#include <torch/extension.h>

// Macros for error checking
#define NCCL_CHECK(cmd) do {                         \
    ncclResult_t r = (cmd);                            \
    if (r != ncclSuccess && r != ncclInProgress) {                          \
        fprintf(stderr, "NCCL Error at %s:%d - %d. Refer to https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t for error code meanings\n",      \
                __FILE__, __LINE__, r);              \
        exit(EXIT_FAILURE);                          \
    }                                                \
} while(0)

#define CUDA_CHECK(cmd) do {                         \
    cudaError_t e = (cmd);                             \
    if (e != cudaSuccess) {                          \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",      \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                          \
    }                                                \
} while(0)

/*
assert_whenever: assertion which ignore whether NDEBUG is set

In C++, assert() is evaluated only when NDEBUG is not set. This is
inconvenient when we want to check the assertion even in release mode.
This macro is a workaround for this problem.
*/

extern "C" {
// Copied from assert.h
extern void __assert_fail (const char *__assertion, const char *__file,
			   unsigned int __line, const char *__function)
     __THROW __attribute__ ((__noreturn__));

#define __ASSERT_FUNCTION	__extension__ __PRETTY_FUNCTION__
#  define assert_whenever(expr)							\
     (static_cast <bool> (expr)						\
      ? void (0)							\
      : __assert_fail (#expr, __FILE__, __LINE__, __ASSERT_FUNCTION))
}

// Functions for conversion between ncclUniqueId and vector
typedef std::vector<uint8_t> ncclUniqueIdVec;
inline ncclUniqueIdVec ncclUniqueId_to_Vec(const ncclUniqueId& id) {
    return ncclUniqueIdVec(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}
inline ncclUniqueId ncclUniqueId_from_Vec(const ncclUniqueIdVec& vec) {
    ncclUniqueId id;
    assert (vec.size() == NCCL_UNIQUE_ID_BYTES);
    std::copy(vec.begin(), vec.end(), id.internal);
    return id;
}

// get_nccl_uniqueid - Get a NCCL unique id for initializing RNCCLComm
ncclUniqueIdVec get_nccl_unique_id() {
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    return ncclUniqueId_to_Vec(id);
}

// RNCCLComm - The communicator object
class RNCCLComm {
public:
    int world_size;
    int total_rank;
    ncclUniqueId unique_id;
    ncclComm_t comm;
    // ncclComm_t *send_comm_list;
    // ncclComm_t *recv_comm_list;
    cudaStream_t stream;
    // cudaStream_t *send_stream_list;
    // cudaStream_t *recv_stream_list;

    // Create a communicator
    RNCCLComm(const ncclUniqueIdVec &vec, int world_size, int total_rank) {
        assert_whenever (total_rank >= 0 && total_rank < world_size);

        this->unique_id = ncclUniqueId_from_Vec(vec);
        this->world_size = world_size;
        this->total_rank = total_rank;

        // Create the NCCL communicator
        // Set NCCL config, including minCTA, maxCTA, here
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking = 1;
        // config.minCTAs = 1;
        // config.maxCTAs = 32;
        // config.cgaClusterSize = 0;
        // config.netName = "Socket";
        NCCL_CHECK(ncclCommInitRankConfig(&(this->comm), world_size, this->unique_id, total_rank, &config));
        // // ncclResult_t state;
        // // do {
        // //     NCCL_CHECK(ncclCommGetAsyncError(this->comm, &state));
        // //     // Handle outside events, timeouts, progress, ...
        // // } while(state == ncclInProgress);

        // this->send_comm_list = new ncclComm_t[world_size];
        // this->recv_comm_list = new ncclComm_t[world_size];

        // for (int src_rank = 0; src_rank < world_size; ++src_rank) {
        //     for (int dst_rank = 0; dst_rank < world_size; ++dst_rank) {
        //         if (src_rank == dst_rank) {
        //             continue;
        //         }
        //         ncclConfig_t p2p_config = NCCL_CONFIG_INITIALIZER;
        //         p2p_config.splitShare = 1;
        //         if (total_rank == src_rank) {
        //             NCCL_CHECK(ncclCommSplit(this->comm, 0, 0, &this->send_comm_list[dst_rank], &p2p_config));
        //         }
        //         else if (total_rank == dst_rank) {
        //             NCCL_CHECK(ncclCommSplit(this->comm, 0, 1, &this->recv_comm_list[src_rank], &p2p_config));
        //         }
        //         else {
        //             ncclComm_t newcomm;
        //             NCCL_CHECK(ncclCommSplit(this->comm, NCCL_SPLIT_NOCOLOR, -1, &newcomm, &p2p_config));
        //         }
        //         // do {
        //         //     NCCL_CHECK(ncclCommGetAsyncError(this->comm, &state));
        //         //     // Handle outside events, timeouts, progress, ...
        //         // } while(state == ncclInProgress);
        //     }
        // }

        // Create the CUDA stream
        // Set stream config, including priority, here
        const int priority = 0;
        CUDA_CHECK(cudaStreamCreateWithPriority(&(this->stream), cudaStreamNonBlocking, priority));

        // this->send_stream_list = new cudaStream_t[world_size];
        // this->recv_stream_list = new cudaStream_t[world_size];
        // for (int i = 0; i < world_size; ++i) {
        //     if (i == total_rank) {
        //         continue;
        //     }
        //     CUDA_CHECK(cudaStreamCreateWithPriority(&(this->send_stream_list[i]), cudaStreamNonBlocking, priority));
        //     CUDA_CHECK(cudaStreamCreateWithPriority(&(this->recv_stream_list[i]), cudaStreamNonBlocking, priority));
        // }
    }

    // Finialize
    ~RNCCLComm() {
        // Destroy the NCCL communicator
        NCCL_CHECK(ncclCommDestroy(this->comm));

        // for (int i = 0; i < world_size; ++i) {
        //     if (i == total_rank) {
        //         continue;
        //     }
        //     if (send_comm_list[i] != NULL) {
        //         NCCL_CHECK(ncclCommDestroy(send_comm_list[i]));
        //     }
        //     if (recv_comm_list[i] != NULL) {
        //         NCCL_CHECK(ncclCommDestroy(recv_comm_list[i]));
        //     }
        // }
        // delete[] send_comm_list;
        // delete[] recv_comm_list;

        // Destroy the CUDA stream
        CUDA_CHECK(cudaStreamDestroy(this->stream));

        // for (int i = 0; i < world_size; ++i) {
        //     if (i == total_rank) {
        //         continue;
        //     }
        //     CUDA_CHECK(cudaStreamDestroy(this->send_stream_list[i]));
        //     CUDA_CHECK(cudaStreamDestroy(this->recv_stream_list[i]));
        // }
    }

    // Group start/end
    void nccl_group_start() {
        NCCL_CHECK(ncclGroupStart());
    }

    void nccl_group_end() {
        NCCL_CHECK(ncclGroupEnd());
        // ncclResult_t state;
        // do {
        //     NCCL_CHECK(ncclCommGetAsyncError(this->comm, &state));
        //     // Handle outside events, timeouts, progress, ...
        // } while(state == ncclInProgress);
    }

    // P2P send/recv
    void nccl_send(torch::Tensor data, int peer) {
        assert_whenever (data.is_contiguous() && data.device().is_cuda());
        assert_whenever (peer >= 0 && peer < this->world_size);
        if (data.numel() == 0) {
            return;
        }
        NCCL_CHECK(ncclSend(
            (char*)data.data_ptr(),
            data.numel()*data.element_size(),
            ncclChar,
            peer,
            this->comm,
            this->stream
            // 1,
            // this->send_comm_list[peer],
            // this->send_stream_list[peer]
        ));
    }

    void nccl_recv(torch::Tensor data, int peer) {
        assert_whenever (data.is_contiguous() && data.device().is_cuda());
        assert_whenever (peer >= 0 && peer < this->world_size);
        if (data.numel() == 0) {
            return;
        }
        NCCL_CHECK(ncclRecv(
            (char*)data.data_ptr(),
            data.numel()*data.element_size(),
            ncclChar,
            peer,
            this->comm,
            this->stream
            // 0,
            // this->recv_comm_list[peer],
            // this->recv_stream_list[peer]
        ));
    }

    // CUDA stream synchronize
    void wait_for_default_stream() {
        // Let the current stream wait for the default stream
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreate(&event));
        CUDA_CHECK(cudaEventRecord(event, nullptr));
        CUDA_CHECK(cudaStreamWaitEvent(this->stream, event, 0));

        // for (int i = 0; i < world_size; ++i) {
        //     if (i == total_rank) {
        //         continue;
        //     }
        //     CUDA_CHECK(cudaStreamWaitEvent(this->send_stream_list[i], event, 0));
        //     CUDA_CHECK(cudaStreamWaitEvent(this->recv_stream_list[i], event, 0));
        // }
        
        CUDA_CHECK(cudaEventDestroy(event));
    }

    void let_default_stream_wait() {
        // Let the default stream wait for the current stream
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreate(&event));
        CUDA_CHECK(cudaEventRecord(event, this->stream));
        CUDA_CHECK(cudaStreamWaitEvent(nullptr, event, 0));
        CUDA_CHECK(cudaEventDestroy(event));

        // for (int i = 0; i < world_size; ++i) {
        //     if (i == total_rank) {
        //         continue;
        //     }
        //     CUDA_CHECK(cudaEventCreate(&event));
        //     CUDA_CHECK(cudaEventRecord(event, this->send_stream_list[i]));
        //     CUDA_CHECK(cudaStreamWaitEvent(nullptr, event, 0));
        //     CUDA_CHECK(cudaEventDestroy(event));

        //     CUDA_CHECK(cudaEventCreate(&event));
        //     CUDA_CHECK(cudaEventRecord(event, this->recv_stream_list[i]));
        //     CUDA_CHECK(cudaStreamWaitEvent(nullptr, event, 0));
        //     CUDA_CHECK(cudaEventDestroy(event));
        // }
    }
};

namespace py = pybind11;

PYBIND11_MODULE(rnccl, m) {
    m.doc() = "Raw NCCL bindings for Python";

    m.def("hello_world", [&]() {printf("Hello world from rnccl!\n"); },
    "A function for verifying the model is correctly loaded");

    m.def("get_nccl_unique_id", &get_nccl_unique_id,
    "Get a NCCL unique id for initializing RNCCLComm");

    py::class_<RNCCLComm>(m, "RNCCLComm")
        .def(py::init<ncclUniqueIdVec, int, int>())
        .def("nccl_group_start", &RNCCLComm::nccl_group_start)
        .def("nccl_group_end", &RNCCLComm::nccl_group_end)
        .def("nccl_send", &RNCCLComm::nccl_send)
        .def("nccl_recv", &RNCCLComm::nccl_recv)
        .def("wait_for_default_stream", &RNCCLComm::wait_for_default_stream)
        .def("let_default_stream_wait", &RNCCLComm::let_default_stream_wait);
}