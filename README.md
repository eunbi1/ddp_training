# ddp_training

₩₩₩
CUDA_VISIBLE_DEVICES=0,1 LOCAL_RANK=0,1 python -m torch.distributed.launch  --master_port 12342 --nproc_per_node 2  ddp.py
₩₩₩
