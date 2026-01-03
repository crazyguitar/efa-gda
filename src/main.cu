#include <gda/affinity.h>
#include <gda/efa.h>
#include <gda/mpi.h>
#include <gda/taskset.h>

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  auto& mpi = MPI::Get();
  auto& loc = GPUloc::Get();
  auto rank = mpi.GetWorldRank();
  auto device = mpi.GetLocalRank();
  auto& affinity = loc.GetGPUAffinity()[device];
  std::vector<EFA> efas;

  Taskset::Set(affinity.cores[device]->logical_index);
  efas.reserve(affinity.efas.size());
  for (auto e : affinity.efas) efas.emplace_back(EFA(e));

  if (rank == 0) {
    std::cout << affinity << std::endl;
  }
}
