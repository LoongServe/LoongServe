#! /bin/bash

# is_master = 
ips=$(ip a)
echo "self ip: $ips"

master_ip='10.140.66.108'
is_master=false

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

if [[ $ips =~ $master_ip ]]; then # 
    echo "is master"
    is_master=true
    ray start --head
else
    echo "not master"
    sleep 2
    ray start --address="$master_ip":6379
fi

sleep 10
ray status

if [[ $ips =~ $master_ip ]]; then
    python -m loongserve.longserve_server.api_server --host $master_ip --port 8400 --model_dir /mnt/petrelfs/wubingyang/intlsy/weights/LWM-Text-1M --tokenizer_mode auto --max_total_token_num 210000 --max_req_input_len 799999 --max_req_total_len 800000 --running_max_req_size 1024 --tp_world_size 2 --sp_world_size 8 --mode _token_decode_attention_overlapped --max_mig_len 10000 --local_world_size 8
else
    sleep 10000000
fi