import torch	# Please "import torch" before importing rnccl
import rnccl

@torch.inference_mode()
def worker_subroutine(world_size, rank, unique_id):
	print(f"Worker {rank=} starts")
	torch.set_default_device(torch.device("cuda", rank))
	torch.cuda.set_device(rank)

	comm = rnccl.RNCCLComm(unique_id, world_size, rank)

	data1 = torch.full((4*1024*1024*1024,), rank, device="cuda")	# A tensor of size 16gb
	data2 = torch.empty_like(data1)
	data3 = torch.randn(10000, 10000)
	data4 = torch.randn(10000, 10000)
	data5 = torch.empty(10000, 10000)
	next_rank = (rank + 1) % world_size
	prev_rank = (rank - 1) % world_size

	def run():
		comm.nccl_group_start()
		comm.nccl_send(data1, next_rank)
		comm.nccl_recv(data2, prev_rank)
		comm.nccl_group_end()

		torch.mm(data3, data4, out=data5)	# This should run concurrently with the communication

		comm.let_default_stream_wait()
	
	for _ in range(3):
		run()
		
	print(f"Worker {rank=} ends")
	print(f"Worker {rank=} received {data2[0].item()}")
	print(f"{data5[0][0].item()+data3[0][0].item()+data4[0][0].item()+data1[0].item()+data2[0].item()}")

if __name__ == "__main__":
	torch.multiprocessing.set_start_method('spawn', force=True)
	world_size = 4
	unique_id = rnccl.get_nccl_unique_id()
	processes = []
	for rank in range(world_size):
		p = torch.multiprocessing.Process(target=worker_subroutine, args=(world_size, rank, unique_id))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
