#include <gda/peer.h>

#include <efa_cuda_dp.cuh>

__global__ void gda_send_kernel(
    efa_cuda_qp* qp,
    efa_cuda_cq* cq,
    uint16_t ah,
    uint32_t remote_qpn,
    uint32_t remote_qkey,
    uint64_t addr,
    uint32_t len,
    uint32_t lkey
) {
  if (threadIdx.x != 0) return;

  uint8_t wr_buf[64] = {0};
  efa_cuda_init_send_wr(wr_buf, 0);
  efa_cuda_wr_set_remote(wr_buf, ah, remote_qpn, remote_qkey);
  efa_cuda_wr_set_sge(wr_buf, lkey, addr, len);

  efa_cuda_start_sq_batch(qp, 1);
  efa_cuda_sq_batch_place_wr(qp, 0, wr_buf);
  efa_cuda_flush_sq_wrs(qp);

  for (int i = 0; i < 1000000; ++i) {
    void* wc = efa_cuda_cq_poll(cq, 0);
    if (wc) {
      efa_cuda_cq_pop(cq, 1);
      printf("GPU: send completed\n");
      return;
    }
  }
  printf("GPU: send timeout\n");
}

__global__ void gda_recv_kernel(efa_cuda_qp* qp, efa_cuda_cq* cq, uint64_t addr, uint32_t len, uint32_t lkey) {
  if (threadIdx.x != 0) return;

  efa_cuda_post_recv_wr(qp, addr, len, lkey);
  efa_cuda_flush_rq_wrs(qp);

  for (int i = 0; i < 1000000; ++i) {
    void* wc = efa_cuda_cq_poll(cq, 0);
    if (wc) {
      uint32_t byte_len = efa_cuda_wc_read_byte_len(wc);
      efa_cuda_cq_pop(cq, 1);
      printf("GPU: recv completed, len=%u\n", byte_len);
      return;
    }
  }
  printf("GPU: recv timeout\n");
}

int main(int argc, char* argv[]) {
  Peer peer;
  peer.Exchange();
  peer.Connect();

  const auto rank = peer.mpi.GetWorldRank();
  const auto size = peer.mpi.GetWorldSize();

  if (size < 2) {
    if (rank == 0) std::cerr << "Need at least 2 ranks\n";
    return 1;
  }

  auto buf = peer.CreateBuffer(4096);
  cudaMemset(buf->Data(), rank + 1, buf->Size());

  uint32_t lkey = peer.GetLkey(*buf);
  auto* efa = peer.GetEFA();

  MPI_Barrier(MPI_COMM_WORLD);

  cudaLaunchConfig_t cfg = {.gridDim = 1, .blockDim = 256};

  if (rank == 0) {
    uint16_t ah, qpn;
    uint32_t qkey;
    peer.QueryPeer(1, 0, &ah, &qpn, &qkey);
    LAUNCH_KERNEL(&cfg, gda_send_kernel, efa->GetGdaQP(), efa->GetGdaSendCQ(), ah, qpn, qkey, (uint64_t)buf->Data(), 64, lkey);
  } else if (rank == 1) {
    LAUNCH_KERNEL(&cfg, gda_recv_kernel, efa->GetGdaQP(), efa->GetGdaRecvCQ(), (uint64_t)buf->Data(), (uint32_t)buf->Size(), lkey);
  }
  cudaDeviceSynchronize();

  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}
