import os 
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp 
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP 

def example(rank, world_size):
    # default process group
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f'rank: {rank} local_rank: {local_rank} world_size: {world_size})')
    dist.init_process_group(backend="nccl", rank= rank, world_size = world_size)
    # 이 함수를 호출함으로써 여러 개의 프로세스가 서로 통신할 수 있도록 초기화한다
    # 만일 초기화를 하지 않는다면 발생하는 문제점
    # 1. 통신 실패.
    # 2. 동기화 문제.
    # 3. 효율성 저하.
    # - backgend
    # 1) gloo: CPU 환경에서 잘 작동함
    # 2) nccl: NVIDIA GPUs에서 최적화됨. 
    # - rank: 현재 프로세스의 순위 혹은 ID이다. 이 순위를 통해 서로를 구분한다. 
    # - world_size: 분산 훈련에 참여하는 프로세스의 총 개수이다. 
  
    # create model
    model = nn.Linear(10,10).to(rank)
    # create ddp model
    model = DDP(model, device_ids=[rank])
    # define loss & optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001)
    
    #forward pass
    outputs = model(torch.randn(20,10).to(rank))
    labels = torch.randn(20,10).to(rank)
    
    #backward pss
    loss_fn(outputs,labels).backward()
    optimizer.step()
    print('forward & backward')
    
def main():
    # global rank 
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f'Local rank is {local_rank} and World size is {world_size}')
    mp.spawn(example, args = (world_size,), nprocs= world_size, join=True)
    
    # mp.spawn 함수는 PyTorch에서 제공하는 멀티 프로세싱(multiprocessing) 유틸리티의 일부로, 
    # 병렬 프로세스를 생성하고 관리하는 데 사용된다. 
    # GPUs init -> mp.spawn 
    # - example: 병렬로 실행될 함수이다. 이 함수는 각각의 별도 프로세스에서 실행된다. 
    # - args: example 함수에 전달될 인자들의 튜플이다. 여기서 (world_size,)는 example 함수에 전달되는 인자로, 전체 프로세스의 수를 나타내는 world_size를 포함한다. 
    # - nprocs: 생성할 프로세스의 총 수입니다. world_size를 이 값으로 설정함으로써, world_size 개수만큼의 프로세스가 생성된다. 
    # - join: 모든 프로세스가 종료될 때까지 기다릴 것인지를 결정하는 불린(Boolean) 값이다. join=True로 설정하면, 모든 프로세스가 종료될 때까지 메인 프로세스의 실행이 대기 상태가 된ㄷ.
    # - mp.spawn은 호출되는 함수에 자동으로 rank 인자를 추가로 전달한다.
    # - 따라서, 호출되는 함수가 추가적인 인자를 필요로 할 경우, 이들을 args 튜플에 포함시켜야 한다.
   

if __name__ == "__main__":
    sys.exit(main())