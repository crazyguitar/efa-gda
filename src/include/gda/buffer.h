/**
 * @file buffer.h
 * @brief GPU buffer for GDA RDMA operations using DMABUF
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <gda/common.h>
#include <gda/efa.h>
#include <rdma/fi_domain.h>
#include <unistd.h>

#include <cstdlib>

/**
 * @brief GPU memory buffer for GDA RDMA operations
 *
 * Allocates GPU memory and exports dmabuf for zero-copy RDMA registration.
 */
class GpuBuffer : private NoCopy {
 public:
  GpuBuffer() = delete;
  GpuBuffer(GpuBuffer&&) = delete;
  GpuBuffer& operator=(GpuBuffer&&) = delete;

  GpuBuffer(fid_domain* domain, int device, size_t size) : domain_{domain}, device_{device}, size_{size} {
    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t alloc_size = ((size + page_size - 1) / page_size) * page_size;

    // Allocate extra for alignment
    CUDA_OK(cudaMalloc(&raw_, alloc_size + page_size - 1));

    // Page-align the data pointer
    data_ = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw_) + page_size - 1) & ~(page_size - 1));
    alloc_size_ = alloc_size;

    CU_OK(cuMemGetHandleForAddressRange(&dmabuf_fd_, (CUdeviceptr)data_, alloc_size_, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));

    struct fi_mr_attr attr{};
    struct fi_mr_dmabuf dmabuf{};
    dmabuf.fd = dmabuf_fd_;
    dmabuf.offset = 0;
    dmabuf.len = size;
    dmabuf.base_addr = data_;

    attr.iov_count = 1;
    attr.access = FI_SEND | FI_RECV | FI_REMOTE_WRITE | FI_REMOTE_READ;
    attr.iface = FI_HMEM_CUDA;
    attr.device.cuda = device;
    attr.dmabuf = &dmabuf;

    FI_CHECK(fi_mr_regattr(domain_, &attr, FI_MR_DMABUF, &mr_));
  }

  ~GpuBuffer() {
    if (mr_) fi_close(&mr_->fid);
    if (dmabuf_fd_ >= 0) close(dmabuf_fd_);
    if (raw_) cudaFree(raw_);
  }

  [[nodiscard]] void* Data() noexcept { return data_; }
  [[nodiscard]] size_t Size() const noexcept { return size_; }
  [[nodiscard]] fid_mr* MR() noexcept { return mr_; }
  [[nodiscard]] uint64_t Key() const noexcept { return fi_mr_key(mr_); }

  template <typename T>
  T* As() noexcept {
    return static_cast<T*>(data_);
  }

 private:
  fid_domain* domain_ = nullptr;
  fid_mr* mr_ = nullptr;
  void* raw_ = nullptr;
  void* data_ = nullptr;
  int device_ = -1;
  int dmabuf_fd_ = -1;
  size_t size_ = 0;
  size_t alloc_size_ = 0;
};
