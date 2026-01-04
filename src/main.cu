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

  printf("GPU send: qp=%p cq=%p ah=%u qpn=%u qkey=%u addr=%p len=%u lkey=%u\n", qp, cq, ah, remote_qpn, remote_qkey, (void*)addr, len, lkey);
  printf("GPU send: sq.buf=%p sq.db=%p sq.pc=%u sq.phase=%d\n", qp->sq.wq.buf, qp->sq.wq.db, qp->sq.wq.pc, qp->sq.wq.phase);

  uint8_t wr_buf[64] = {0};
  efa_cuda_init_send_wr(wr_buf, 0);
  efa_cuda_wr_set_remote(wr_buf, ah, remote_qpn, remote_qkey);
  efa_cuda_wr_set_sge(wr_buf, lkey, addr, len);

  // Debug: print WQE bytes
  printf("GPU send: WQE bytes: ");
  for (int i = 0; i < 64; i++) printf("%02x ", wr_buf[i]);
  printf("\n");

  efa_cuda_start_sq_batch(qp, 1);
  efa_cuda_sq_batch_place_wr(qp, 0, wr_buf);
  printf("GPU send: after place_wr, sq.pc=%u sq.wqes_pending=%u\n", qp->sq.wq.pc, qp->sq.wq.wqes_pending);

  // Use library flush function
  efa_cuda_flush_sq_wrs(qp);
  printf("GPU send: after flush, sq.pc=%u\n", qp->sq.wq.pc);

  for (int i = 0; i < 5000000; ++i) {  // Increased timeout
    void* wc = efa_cuda_cq_poll(cq, 0);
    if (wc) {
      efa_cuda_cq_pop(cq, 1);
      printf("GPU: send completed\n");
      return;
    }
    if (i == 0) {
      // Print CQ info on first poll
      printf("GPU send poll: cq.buf=%p cq.cc=%u cq.phase=%d cq.num_entries=%u\n", cq->buf, cq->cc, cq->phase, cq->num_entries);
      uint8_t* buf = cq->buf;
      printf(
          "GPU send poll: CQE[0] bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n", buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]
      );
    }
  }
  printf("GPU: send timeout, cq.cc=%u cq.phase=%d\n", cq->cc, cq->phase);
}

__global__ void gda_post_recv_kernel(efa_cuda_qp* qp, uint64_t addr, uint32_t len, uint32_t lkey) {
  if (threadIdx.x != 0) return;
  printf("GPU recv: posting rq.buf=%p rq.db=%p rq.pc=%u addr=%p len=%u lkey=%u\n", qp->rq.wq.buf, qp->rq.wq.db, qp->rq.wq.pc, (void*)addr, len, lkey);
  efa_cuda_post_recv_wr(qp, addr, len, lkey);

  // Debug: print RQ buffer content
  uint8_t* rq_buf = qp->rq.wq.buf;
  printf("GPU recv: RQ[0] bytes: ");
  for (int i = 0; i < 16; i++) printf("%02x ", rq_buf[i]);
  printf("\n");

  efa_cuda_flush_rq_wrs(qp);
  printf("GPU recv: after flush, rq.pc=%u db_val=%u\n", qp->rq.wq.pc, *qp->rq.wq.db);
}

__global__ void gda_poll_recv_kernel(efa_cuda_cq* cq) {
  if (threadIdx.x != 0) return;
  printf("GPU poll_recv: cq=%p cq.buf=%p cq.cc=%u cq.phase=%d entry_size=%u\n", cq, cq->buf, cq->cc, cq->phase, cq->entry_size);

  // Print first CQE bytes
  uint8_t* buf = cq->buf;
  printf("GPU poll_recv: CQE[0] bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n", buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]);

  for (int i = 0; i < 10000000; ++i) {
    void* wc = efa_cuda_cq_poll(cq, 0);
    if (wc) {
      uint32_t byte_len = efa_cuda_wc_read_byte_len(wc);
      efa_cuda_cq_pop(cq, 1);
      printf("GPU: recv completed, len=%u\n", byte_len);
      return;
    }
    if (i == 5000000) {
      // Mid-poll check
      printf(
          "GPU poll_recv mid: CQE[0] bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n", buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]
      );
    }
  }
  printf("GPU: recv timeout, cq.cc=%u cq.phase=%d\n", cq->cc, cq->phase);
  printf(
      "GPU poll_recv end: CQE[0] bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n", buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]
  );
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

  // Rank 1: post recv first
  if (rank == 1) {
    LAUNCH_KERNEL(&cfg, gda_post_recv_kernel, efa->GetGdaQP(), (uint64_t)buf->Data(), (uint32_t)buf->Size(), lkey);
    cudaDeviceSynchronize();
  }

  MPI_Barrier(MPI_COMM_WORLD);  // Ensure recv is posted before send

  // Rank 0: send
  if (rank == 0) {
    uint16_t ah;
    uint32_t qpn, qkey;
    peer.QueryPeer(1, 0, &ah, &qpn, &qkey);
    printf("Sending to ah=%u qpn=%u qkey=%u lkey=%u\n", ah, qpn, qkey, lkey);
    LAUNCH_KERNEL(&cfg, gda_send_kernel, efa->GetGdaQP(), efa->GetGdaSendCQ(), ah, qpn, qkey, (uint64_t)buf->Data(), 64, lkey);
    cudaDeviceSynchronize();
    // Debug: check CQ from CPU after timeout
    efa->DebugCQFromCPU("TX_CQ", efa->GetSendCQBuf());
  }

  // Rank 1: poll for recv completion
  if (rank == 1) {
    LAUNCH_KERNEL(&cfg, gda_poll_recv_kernel, efa->GetGdaRecvCQ());
    cudaDeviceSynchronize();
    // Debug: check CQ from CPU after timeout
    efa->DebugCQFromCPU("RX_CQ", efa->GetRecvCQBuf());
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}
