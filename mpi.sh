#!/bin/bash

PROGRAM="$0"
IMAGE=""
GPUS=8
NAME="mpi-container"
FORCE=0

err() {
  echo -e "[$(date +'%Y-%m-%dT%H:%M:%S%z')][error] $*" >&2
}

usage() {
  set +x
  cat <<EOF
usage:  $PROGRAM [OPTIONS] params

options:

  -h,--help             show this help
  -i,--image [image]    docker image for the experiment
  -g,--gpus [N]         number of gpus
  -n,--name [NAME]      container name
  -f,--force            force relaunch containers
EOF
}

check_containers_running() {
  local name="${1}"
  local num_nodes
  local running_count

  num_nodes=$(scontrol show hostnames | wc -l)
  running_count=$(srun bash -c "docker ps -q -f name=${name} | wc -l" 2>/dev/null | awk '{s+=$1} END {print s}')

  [[ "${running_count}" -eq "${num_nodes}" ]]
}

launch_container() {
  local image="${1}"
  local name="${2}"
  local ssh_home="${HOME}/.ssh"
  local cmd

  cmd="$(cat <<EOF
set -ex
TOKEN="\$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")"
INSTANCE_TYPE="\$(curl -sS -H "X-aws-ec2-metadata-token: \$TOKEN" http://169.254.169.254/latest/meta-data/instance-type)"
echo \$INSTANCE_TYPE
IMAGE="${image}"
SSH_HOME="${ssh_home}"
device=("--device=/dev/gdrdrv")
while IFS= read -r -d '' d; do
  device+=("--device=\${d}")
done < <(find "/dev/infiniband" -name "uverbs*" -print0)
FSX_MOUNT="/fsx/:/fsx"
docker run --gpus ${GPUS} \
  --privileged \
  --rm \
  -d \
  --name "${name}" \
  --uts=host \
  --ulimit stack=67108864 \
  --ulimit memlock=-1 \
  --ipc=host --net=host \
  --security-opt seccomp=unconfined \
  -e RDMAV_FORK_SAFE=1 \
  -v "\${FSX_MOUNT}" \
  -v "\${SSH_HOME}:/root/.ssh" \
  "\${device[@]}" \
  "${image}" \
  sleep infinity

docker exec -t "${name}" /bin/bash -c "printf 'Port 2022\n' >> /etc/ssh/sshd_config"
docker exec -t "${name}" /bin/bash -c "service ssh start"
EOF
)"

  if ! srun bash -c "${cmd}"; then
    err "launch container ${image} fail"
    return 1
  fi
}

launch() {
  local name="${1}"
  local args=("${@:2}")
  local arr
  local hosts
  local cmd

  mapfile -t arr < <(scontrol show hostnames | sort)
  OLDIFS="${IFS}"
  IFS=","
  hosts="${arr[*]}"
  IFS="${OLDIFS}"

  cmd="$(cat <<EOF
docker exec -t "${name}" mpirun \
  -N "${GPUS}" \
  --allow-run-as-root \
  --host "${hosts}" \
  --mca plm_rsh_no_tree_spawn 1 \
  --mca plm_rsh_num_concurrent "${#arr[@]}" \
  --mca plm_rsh_args "-p 2022" \
  --mca pml ^ucx \
  --mca btl tcp,self \
  --mca btl_tcp_if_exclude lo,docker0,veth_def_agent \
  --oversubscribe \
  --tag-output \
  -x LD_LIBRARY_PATH \
  ${args[@]}
EOF
)"

  srun -N 1 bash -c "${cmd}"
}

clean() {
  local name="${1}"
  local cmd
  cmd=$(cat <<EOF
docker rm -f ${name} 2>/dev/null || true
EOF
)

  if ! srun bash -c "${cmd}"; then
    err "clean docker container fail"
    return 1
  fi
}

run() {
  set -exo pipefail

  local image="${1}"
  local name="${2}"
  local force="${3}"
  local cmd=("${@:4}")

  if [[ "${force}" -eq 1 ]]; then
    clean "${name}"
  fi

  if ! check_containers_running "${name}"; then
    if ! launch_container "${image}" "${name}"; then
      return 1
    fi
  fi

  if ! launch "${name}" "${cmd[@]}"; then
    err "launch ${cmd[*]} fail"
    return 1
  fi
}

while (( "$#" )); do
  case "$1" in
    -h|-\?|--help) usage; exit 0 ;;
    -i|--image) IMAGE="${2}"; shift 2 ;;
    -g|--gpus) GPUS="${2}"; shift 2 ;;
    -n|--name) NAME="${2}"; shift 2 ;;
    -f|--force) FORCE=1; shift ;;
    *) break
  esac
done

if [[ -z "${IMAGE}" ]]; then
  err "image is required (-i)"
  usage
  exit 1
fi

if ! run "${IMAGE}" "${NAME}" "${FORCE}" "$@"; then
  err "run fail"
  exit 1
fi
