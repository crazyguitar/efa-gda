/**
 * @file efa.h
 * @brief EFA (Elastic Fabric Adapter) fabric initialization and management
 */
#pragma once

#include <cuda.h>
#include <efa_cuda_dp.h>
#include <gda/common.h>
#include <hwloc.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_ext_efa.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <utility>

/** @brief Maximum buffer size for endpoint addresses */
static constexpr size_t kMaxAddrSize = 64;
/** @brief Actual size of EFA endpoint addresses */
static constexpr size_t kAddrSize = 32;

/**
 * @brief Check libfabric call result and throw on error
 * @param exp Libfabric expression to evaluate
 * @throws std::runtime_error if expression returns non-zero
 */
#define FI_CHECK(exp)                                                            \
  do {                                                                           \
    auto rc = exp;                                                               \
    if (rc) {                                                                    \
      auto msg = fmt::format(#exp " fail. error({}): {}", rc, fi_strerror(-rc)); \
      SPDLOG_ERROR(msg);                                                         \
      throw std::runtime_error(msg);                                             \
    }                                                                            \
  } while (0)

/**
 * @brief Check libfabric call result matches expected value
 * @param exp Libfabric expression to evaluate
 * @param expect Expected return value
 * @throws std::runtime_error if expression result != expect
 */
#define FI_EXPECT(exp, expect)                                                   \
  do {                                                                           \
    auto rc = (exp);                                                             \
    if (rc != expect) {                                                          \
      auto msg = fmt::format(#exp " fail. error({}): {}", rc, fi_strerror(-rc)); \
      SPDLOG_ERROR(msg);                                                         \
      throw std::runtime_error(msg);                                             \
    }                                                                            \
  } while (0)

/**
 * @brief Singleton for EFA provider information discovery
 *
 * Queries libfabric for available EFA providers with required capabilities
 * (FI_MSG, FI_RMA, FI_HMEM). Thread-safe singleton pattern.
 */
class FI : private NoCopy {
 public:
  /**
   * @brief Get singleton fi_info for EFA providers
   * @return Linked list of fi_info structures for available EFA devices
   */
  static const struct fi_info* Get() {
    static FI instance;
    return instance.info_;
  }

 private:
  FI() : info_{New()} {}
  FI(FI&&) = delete;
  FI& operator=(FI&&) = delete;

  ~FI() {
    if (info_) {
      fi_freeinfo(info_);
      info_ = nullptr;
    }
  }

 private:
  static struct fi_info* New() {
    int rc = 0;
    struct fi_info* hints = nullptr;
    struct fi_info* info = nullptr;
    hints = fi_allocinfo();
    if (!hints) {
      SPDLOG_ERROR("fi_allocinfo fail.");
      goto end;
    }

    hints->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->fabric_attr->prov_name = strdup("efa");
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->domain_attr->threading = FI_THREAD_SAFE;
    hints->domain_attr->progress = FI_PROGRESS_MANUAL;
    hints->mode |= FI_CONTEXT | FI_CONTEXT2;

    rc = fi_getinfo(FI_VERSION(2, 3), NULL, NULL, 0, hints, &info);
    if (rc != 0) {
      SPDLOG_ERROR("fi_getinfo fail. error({}): {}", rc, fi_strerror(-rc));
      goto error;
    } else {
      goto end;
    }

  error:
    if (info) {
      fi_freeinfo(info);
      info = nullptr;
    }

  end:
    if (hints) {
      fi_freeinfo(hints);
      hints = nullptr;
    }
    return info;
  }

 private:
  /**
   * @brief Stream output operator for EFA information
   * @param os Output stream
   * @param efa EFA instance to output
   * @return Reference to output stream
   */
  friend std::ostream& operator<<(std::ostream& os, const FI& efa) {
    for (auto cur = efa.info_; !!cur; cur = cur->next) {
      os << fmt::format("provider: {}\n", cur->fabric_attr->prov_name);
      os << fmt::format("  fabric: {}\n", cur->fabric_attr->name);
      os << fmt::format("  domain: {}\n", cur->domain_attr->name);
      os << fmt::format("  version: {}.{}\n", FI_MAJOR(cur->fabric_attr->prov_version), FI_MINOR(cur->fabric_attr->prov_version));
      os << fmt::format("  type: {}\n", fi_tostr(&cur->ep_attr->type, FI_TYPE_EP_TYPE));
      os << fmt::format("  protocol: {}\n", fi_tostr(&cur->ep_attr->protocol, FI_TYPE_PROTOCOL));
    }
    return os;
  }

 private:
  struct fi_info* info_ = nullptr;
};

/**
 * @brief EFA (Elastic Fabric Adapter) wrapper for libfabric operations
 *
 * Manages libfabric resources including fabric, domain, endpoint, completion queue,
 * and address vector. Supports move semantics for efficient resource transfer.
 */
class EFA : private NoCopy {
 public:
  EFA() = delete;
  EFA(EFA&&) = delete;
  EFA& operator=(EFA&&) = delete;

  /**
   * @brief Construct EFA instance for specific hardware device
   * @param efa hwloc object representing the EFA device
   */
  EFA(hwloc_obj_t efa) {
    efa_ = Get(efa);
    ASSERT(!!efa_);
    Open(efa_);
  }

  ~EFA() noexcept {
    // Destroy CUDA resources first
    if (gda_send_cq_) efa_cuda_destroy_cq(gda_send_cq_);
    if (gda_recv_cq_) efa_cuda_destroy_cq(gda_recv_cq_);
    if (gda_qp_) efa_cuda_destroy_qp(gda_qp_);
    // Unregister host-mapped memory
    if (sq_host_buf_) cuMemHostUnregister(sq_host_buf_);
    if (sq_host_db_) cuMemHostUnregister(sq_host_db_);
    if (rq_host_buf_) cuMemHostUnregister(rq_host_buf_);
    if (rq_host_db_) cuMemHostUnregister(rq_host_db_);
    // Close libfabric resources
    if (gda_ep_) fi_close((fid_t)gda_ep_);
    if (txcq_) fi_close((fid_t)txcq_);
    if (rxcq_) fi_close((fid_t)rxcq_);
    if (av_) fi_close((fid_t)av_);
    if (domain_) fi_close((fid_t)domain_);
    if (fabric_) fi_close((fid_t)fabric_);
    // Free GPU CQ buffers and close dmabuf fds
    if (send_cq_dmabuf_fd_ >= 0) close(send_cq_dmabuf_fd_);
    if (recv_cq_dmabuf_fd_ >= 0) close(recv_cq_dmabuf_fd_);
    if (send_cq_buf_) cudaFree(send_cq_buf_);
    if (recv_cq_buf_) cudaFree(recv_cq_buf_);
  }

  const char* GetAddr() const noexcept { return addr_; }
  [[nodiscard]] struct fid_av* GetAV() noexcept { return av_; }
  [[nodiscard]] struct fid_domain* GetDomain() noexcept { return domain_; }
  [[nodiscard]] struct fid_ep* GetEP() noexcept { return gda_ep_; }
  [[nodiscard]] const struct fi_info* GetInfo() const noexcept { return efa_; }
  [[nodiscard]] struct efa_cuda_qp* GetGdaQP() noexcept { return gda_qp_; }
  [[nodiscard]] struct efa_cuda_cq* GetGdaSendCQ() noexcept { return gda_send_cq_; }
  [[nodiscard]] struct efa_cuda_cq* GetGdaRecvCQ() noexcept { return gda_recv_cq_; }

  int QueryAddr(fi_addr_t addr, uint16_t* ahn, uint32_t* remote_qpn, uint32_t* remote_qkey) {
    uint16_t qpn16;
    int ret = gda_ops_->query_addr(gda_ep_, addr, ahn, &qpn16, remote_qkey);
    *remote_qpn = qpn16;
    return ret;
  }

  uint64_t GetMRLkey(struct fid_mr* mr) { return gda_ops_->get_mr_lkey(mr); }

  static std::string Addr2Str(const char* addr) {
    std::string out;
    for (size_t i = 0; i < kAddrSize; ++i) out += fmt::format("{:02x}", addr[i]);
    return out;
  }

  static void Str2Addr(const std::string& addr, char* bytes) noexcept {
    for (size_t i = 0; i < kAddrSize; ++i) sscanf(addr.c_str() + 2 * i, "%02hhx", &bytes[i]);
  }

 private:
  struct fi_info* Get(hwloc_obj_t efa) {
    auto* info = FI::Get();
    for (auto p = info; !!p; p = p->next) {
      ASSERT(!!p->nic);
      ASSERT(p->nic->bus_attr and p->nic->bus_attr->bus_type == FI_BUS_PCI);
      auto fi = p->nic->bus_attr->attr.pci;
      auto hw = efa->attr->pcidev;
      if (fi.domain_id == hw.domain and fi.bus_id == hw.bus and fi.device_id == hw.dev and fi.function_id == hw.func) {
        return const_cast<struct fi_info*>(p);
      }
    }
    return nullptr;
  }

  void Open(struct fi_info* info) {
    CheckNvidiaPeerMappingOverride();

    struct fi_av_attr av_attr{};
    ;
    FI_CHECK(fi_fabric(info->fabric_attr, &fabric_, nullptr));
    FI_CHECK(fi_domain(fabric_, info, &domain_, nullptr));
    FI_CHECK(fi_open_ops(&domain_->fid, FI_EFA_GDA_OPS, 0, (void**)&gda_ops_, nullptr));

    // Create extended CQs for GDA
    CreateExtCQ(info->tx_attr->size, &txcq_, &gda_send_cq_, &send_cq_buf_, &send_cq_dmabuf_fd_);
    CreateExtCQ(info->rx_attr->size, &rxcq_, &gda_recv_cq_, &recv_cq_buf_, &recv_cq_dmabuf_fd_);

    FI_CHECK(fi_av_open(domain_, &av_attr, &av_, nullptr));
    FI_CHECK(fi_endpoint(domain_, info, &gda_ep_, nullptr));
    FI_CHECK(fi_ep_bind(gda_ep_, &txcq_->fid, FI_SEND));
    FI_CHECK(fi_ep_bind(gda_ep_, &rxcq_->fid, FI_RECV));
    FI_CHECK(fi_ep_bind(gda_ep_, &av_->fid, 0));
    FI_CHECK(fi_enable(gda_ep_));

    size_t len = sizeof(addr_);
    FI_CHECK(fi_getname(&gda_ep_->fid, addr_, &len));

    CreateGdaQP();
  }

  void CreateExtCQ(size_t cq_entries, struct fid_cq** cq, struct efa_cuda_cq** gda_cq, void** cq_buf_out, int* dmabuf_fd_out) {
    constexpr uint32_t kEntrySize = 32;
    constexpr uint32_t kAdditionalSpace = 4096;
    const size_t page_size = sysconf(_SC_PAGESIZE);
    const size_t raw_size = cq_entries * kEntrySize + kAdditionalSpace;

    void* cq_buffer = nullptr;
    CUDA_OK(cudaMalloc(&cq_buffer, raw_size));

    CUdeviceptr base_addr = 0;
    size_t total_size = 0;
    CU_OK(cuMemGetAddressRange(&base_addr, &total_size, (CUdeviceptr)cq_buffer));

    CUdeviceptr aligned_ptr = base_addr & ~(page_size - 1);
    size_t aligned_size = ((base_addr + total_size + page_size - 1) & ~(page_size - 1)) - aligned_ptr;

    int dmabuf_fd = -1;
    CU_OK(cuMemGetHandleForAddressRange(&dmabuf_fd, aligned_ptr, aligned_size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));

    uint64_t dmabuf_offset = (uintptr_t)cq_buffer - aligned_ptr;

    struct fi_cq_attr cq_attr{};
    cq_attr.format = FI_CQ_FORMAT_MSG;
    cq_attr.wait_obj = FI_WAIT_NONE;
    cq_attr.size = cq_entries;

    struct fi_efa_cq_init_attr efa_cq_init_attr{};
    efa_cq_init_attr.flags = FI_EFA_CQ_INIT_FLAGS_EXT_MEM_DMABUF;
    efa_cq_init_attr.ext_mem_dmabuf.length = raw_size;
    efa_cq_init_attr.ext_mem_dmabuf.offset = dmabuf_offset;
    efa_cq_init_attr.ext_mem_dmabuf.fd = dmabuf_fd;

    FI_CHECK(gda_ops_->cq_open_ext(domain_, &cq_attr, &efa_cq_init_attr, cq, nullptr));

    struct fi_efa_cq_attr cq_ext_attr{};
    FI_CHECK(gda_ops_->query_cq(*cq, &cq_ext_attr));

    struct efa_cuda_cq_attrs cuda_cq_attrs{};
    cuda_cq_attrs.buffer = static_cast<uint8_t*>(cq_buffer);
    cuda_cq_attrs.num_entries = cq_ext_attr.num_entries;
    cuda_cq_attrs.entry_size = cq_ext_attr.entry_size;

    *gda_cq = efa_cuda_create_cq(&cuda_cq_attrs, sizeof(cuda_cq_attrs));
    ASSERT(*gda_cq);

    *cq_buf_out = cq_buffer;
    *dmabuf_fd_out = dmabuf_fd;
  }

  void CreateGdaQP() {
    struct fi_efa_wq_attr sq_attr{}, rq_attr{};
    FI_CHECK(gda_ops_->query_qp_wqs(gda_ep_, &sq_attr, &rq_attr));

    sq_host_buf_ = sq_attr.buffer;
    sq_host_db_ = sq_attr.doorbell;
    rq_host_buf_ = rq_attr.buffer;
    rq_host_db_ = rq_attr.doorbell;

    void* sq_ptr = nullptr;
    CU_OK(cuMemHostRegister(sq_attr.buffer, sq_attr.num_entries * sq_attr.entry_size, CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP));
    CU_OK(cuMemHostGetDevicePointer((CUdeviceptr*)&sq_ptr, sq_attr.buffer, 0));

    uint32_t* sq_db = nullptr;
    CU_OK(cuMemHostRegister(sq_attr.doorbell, 4, CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP));
    CU_OK(cuMemHostGetDevicePointer((CUdeviceptr*)&sq_db, sq_attr.doorbell, 0));

    void* rq_ptr = nullptr;
    CU_OK(cuMemHostRegister(rq_attr.buffer, rq_attr.num_entries * rq_attr.entry_size, CU_MEMHOSTREGISTER_DEVICEMAP));
    CU_OK(cuMemHostGetDevicePointer((CUdeviceptr*)&rq_ptr, rq_attr.buffer, 0));

    uint32_t* rq_db = nullptr;
    CU_OK(cuMemHostRegister(rq_attr.doorbell, 4, CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP));
    CU_OK(cuMemHostGetDevicePointer((CUdeviceptr*)&rq_db, rq_attr.doorbell, 0));

    struct efa_cuda_qp_attrs qp_attrs{};
    qp_attrs.sq_buffer = static_cast<uint8_t*>(sq_ptr);
    qp_attrs.rq_buffer = static_cast<uint8_t*>(rq_ptr);
    qp_attrs.sq_doorbell = sq_db;
    qp_attrs.rq_doorbell = rq_db;
    qp_attrs.sq_num_entries = sq_attr.num_entries;
    qp_attrs.sq_entry_size = sq_attr.entry_size;
    qp_attrs.sq_max_batch = sq_attr.max_batch;
    qp_attrs.rq_num_entries = rq_attr.num_entries;
    qp_attrs.rq_entry_size = rq_attr.entry_size;

    gda_qp_ = efa_cuda_create_qp(&qp_attrs, sizeof(qp_attrs));
    ASSERT(gda_qp_);
  }

 private:
  friend std::ostream& operator<<(std::ostream& os, const EFA& efa) {
    os << fmt::format("provider: {}\n", efa.efa_->fabric_attr->prov_name);
    os << fmt::format("  fabric: {}\n", efa.efa_->fabric_attr->name);
    os << fmt::format("  domain: {}\n", efa.efa_->domain_attr->name);
    return os;
  }

  struct fi_info* efa_ = nullptr;
  struct fid_fabric* fabric_ = nullptr;
  struct fid_domain* domain_ = nullptr;
  struct fid_ep* gda_ep_ = nullptr;
  struct fid_cq* txcq_ = nullptr;
  struct fid_cq* rxcq_ = nullptr;
  struct fid_av* av_ = nullptr;
  struct fi_efa_ops_gda* gda_ops_ = nullptr;
  struct efa_cuda_cq* gda_send_cq_ = nullptr;
  struct efa_cuda_cq* gda_recv_cq_ = nullptr;
  struct efa_cuda_qp* gda_qp_ = nullptr;
  // CQ GPU buffers and dmabuf fds
  void* send_cq_buf_ = nullptr;
  void* recv_cq_buf_ = nullptr;
  int send_cq_dmabuf_fd_ = -1;
  int recv_cq_dmabuf_fd_ = -1;
  // Host pointers for cuMemHostUnregister
  void* sq_host_buf_ = nullptr;
  void* sq_host_db_ = nullptr;
  void* rq_host_buf_ = nullptr;
  void* rq_host_db_ = nullptr;
  char addr_[kMaxAddrSize] = {0};
};
