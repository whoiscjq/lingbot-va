GPU_ID=${GPU_ID:-0}
START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}


save_root="visualization/inject_two_bowls_for_three_bowls"

mkdir -p $save_root

CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root \


