#!/bin/bash
# full_mesh_ib_write_bw.sh - sequential pairwise mesh

NODES=(192.168.100.10 192.168.100.11 192.168.100.12 192.168.100.13)
DEV=rocep1s0f0
PORT=1
GID=3          # RoCEv2 GID index; check with show_gids
SIZE=1048576   # 1M message; or use -a to sweep
SSH="ssh -o StrictHostKeyChecking=no"

for srv in "${NODES[@]}"; do
  for cli in "${NODES[@]}"; do
    [ "$srv" = "$cli" ] && continue

    echo "=== server=$srv  client=$cli ==="

    # start server in background
    $SSH "$srv" "ib_write_bw -d $DEV -i $PORT -F --report_gbits -x $GID -s $SIZE" &
    sleep 2   # let server bind

    # run client, capture the result line
    $SSH "$cli" "ib_write_bw -d $DEV -i $PORT -F --report_gbits -x $GID -s $SIZE $srv" \
        | tee "result_${srv}_${cli}.txt"

    wait   # server exits after the client disconnects
    sleep 1
  done
done
