/**
 * @file peer.h
 * @brief GDA Peer class for GPU Direct Async EFA communication
 */
#pragma once

#include <gda/affinity.h>
#include <gda/buffer.h>
#include <gda/efa.h>
#include <gda/mpi.h>
#include <gda/taskset.h>

#include <array>
#include <memory>
#include <vector>

/**
 * @brief GDA Peer for GPU Direct Async EFA communication
 *
 * Manages MPI bootstrap, EFA endpoints with GDA support, and peer connections.
 */
class Peer : private NoCopy {
 public:
  using AddrBuffer = std::array<char, kMaxAddrSize>;

  MPI& mpi;
  const GPUloc& loc;
  int device = -1;
  std::vector<std::unique_ptr<EFA>> efas;
  std::vector<std::vector<AddrBuffer>> addrs;
  std::vector<std::vector<fi_addr_t>> fi_addrs;

  Peer() : mpi(MPI::Get()), loc(GPUloc::Get()) {
    device = mpi.GetLocalRank();
    cudaSetDevice(device);

    auto& affinity = loc.GetGPUAffinity()[device];
    Taskset::Set(affinity.cores[device]->logical_index);

    if (mpi.GetWorldRank() == 0) {
      std::cout << fmt::format("CUDA Device {}: \"{}\"\n", device, affinity.prop.name);
      std::cout << affinity << std::flush;
    }

    addrs.resize(mpi.GetWorldSize());
    fi_addrs.resize(mpi.GetWorldSize());
    efas.reserve(affinity.efas.size());
    for (auto e : affinity.efas) efas.emplace_back(std::make_unique<EFA>(e));
  }

  /** @brief Exchange EFA addresses across all ranks via MPI_Allgather */
  void Exchange() {
    const auto world_size = mpi.GetWorldSize();
    std::vector<char> recvbuf(world_size * kMaxAddrSize);

    for (const auto& e : efas) {
      MPI_Allgather(e->GetAddr(), kMaxAddrSize, MPI_BYTE, recvbuf.data(), kMaxAddrSize, MPI_BYTE, MPI_COMM_WORLD);
      for (int r = 0; r < world_size; ++r) {
        AddrBuffer addr{};
        std::memcpy(addr.data(), recvbuf.data() + r * kMaxAddrSize, kMaxAddrSize);
        addrs[r].push_back(std::move(addr));
      }
    }
  }

  /** @brief Insert remote addresses into AV and query GDA info */
  void Connect() {
    const auto world_size = mpi.GetWorldSize();
    const auto rank = mpi.GetWorldRank();

    for (int r = 0; r < world_size; ++r) {
      if (r == rank) continue;
      auto& remotes = addrs[r];
      const auto n = std::min(remotes.size(), efas.size());
      fi_addrs[r].resize(n);
      for (size_t i = 0; i < n; ++i) {
        FI_EXPECT(fi_av_insert(efas[i]->GetAV(), remotes[i].data(), 1, &fi_addrs[r][i], 0, nullptr), 1);
      }
    }
  }

  /** @brief Get GDA QP info for a peer */
  void QueryPeer(int peer, int efa_idx, uint16_t* ah, uint32_t* qpn, uint32_t* qkey) {
    efas[efa_idx]->QueryAddr(fi_addrs[peer][efa_idx], ah, qpn, qkey);
  }

  /** @brief Create GPU buffer with DMABUF registration */
  std::unique_ptr<GpuBuffer> CreateBuffer(size_t size, int efa_idx = 0) {
    return std::make_unique<GpuBuffer>(efas[efa_idx]->GetDomain(), device, size);
  }

  /** @brief Get lkey for a buffer's MR */
  uint32_t GetLkey(GpuBuffer& buf, int efa_idx = 0) { return efas[efa_idx]->GetMRLkey(buf.MR()); }

  EFA* GetEFA(int idx = 0) { return efas[idx].get(); }
  size_t GetNumEFAs() const { return efas.size(); }
};
