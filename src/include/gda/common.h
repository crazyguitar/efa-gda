/**
 * @file common.h
 * @brief Common utilities and base classes for I/O library
 */
#pragma once
#include <spdlog/spdlog.h>

#define ASSERT(exp)                                                           \
  do {                                                                        \
    if (!(exp)) {                                                             \
      SPDLOG_CRITICAL("[{}:{}] " #exp " assertion fail", __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

#define CUDA_OK(exp)                                                                                       \
  do {                                                                                                     \
    cudaError_t err = (exp);                                                                               \
    if (err != cudaSuccess) {                                                                              \
      SPDLOG_CRITICAL("[{}:{}] " #exp " got CUDA error: {}", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                             \
    }                                                                                                      \
  } while (0)

#define CU_OK(exp)                                                                                               \
  do {                                                                                                           \
    CUresult rc = (exp);                                                                                         \
    if (rc != CUDA_SUCCESS) {                                                                                    \
      const char* err_str = nullptr;                                                                             \
      cuGetErrorString(rc, &err_str);                                                                            \
      SPDLOG_ERROR("{} failed with {} ({})", #exp, static_cast<int>(rc), (err_str ? err_str : "Unknown error")); \
      exit(1);                                                                                                   \
    }                                                                                                            \
  } while (0)

#define LAUNCH_KERNEL(cfg, kernel, ...) CUDA_OK(cudaLaunchKernelEx(cfg, kernel, ##__VA_ARGS__))

/**
 * @brief Base class preventing copy operations
 */
struct NoCopy {
 protected:
  NoCopy() = default;
  ~NoCopy() = default;
  NoCopy(NoCopy&&) = default;
  NoCopy& operator=(NoCopy&&) = default;
  NoCopy(const NoCopy&) = delete;
  NoCopy& operator=(const NoCopy&) = delete;
};
