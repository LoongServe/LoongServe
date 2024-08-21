"""
A tiny load balancer used for testing LightLLM + Data Parallel on multiple nodes.

It uses a round-robin strategy to distribute the incoming requests to the available workers.
"""

import argparse
import asyncio

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="A tiny load balancer used for testing LightLLM + Data Parallel on multiple nodes.")
	parser.add_argument(
		"--host",
		type=str,
		default="0.0.0.0"
	)
	parser.add_argument(
		"--port",
		type=int,
		default=8800
	)
	parser.add_argument(
		"--workers",
		type=str,
		nargs="+",
		help="List of workers' address, separated by space"
	)
	args = parser.parse_args()

	async def main():
		next_worker = 0
		num_workers = len(args.workers)

		async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
			nonlocal next_worker
			worker_addr = args.workers[next_worker]
			worker_host, worker_port = worker_addr.split(":")
			next_worker = (next_worker + 1) % num_workers
			print("{} -> {}"
              .format(writer.transport.get_extra_info('peername'),
                      f"{worker_host}:{worker_port}"))

			# Connect to the worker
			worker_reader, worker_writer = await asyncio.open_connection(worker_host, worker_port)

			# Forward data between the client and the worker
			async def proxy_bytes(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
				while True:
					try:
						buf = await reader.read(16384)
						if not buf:
							break
						writer.write(buf)
					except asyncio.TimeoutError:
						break
					except (KeyboardInterrupt, Exception):
						print("Closing proxy")
						await writer.drain()
						writer.close()
						break

			await asyncio.gather(
				proxy_bytes(reader, worker_writer),
				proxy_bytes(worker_reader, writer)
			)

			
		server = await asyncio.start_server(
			handle_client,
			args.host,
			args.port
		)

		addr = server.sockets[0].getsockname()
		print(f"Serving on {addr}")

		async with server:
			await server.serve_forever()
		
	asyncio.run(main())
