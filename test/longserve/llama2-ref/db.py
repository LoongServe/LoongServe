import sqlite3

class RecordManager:
	"""
	To speed up our experiment, we store previous experiment results in a database.
	"""
	def __init__(
		self,
		filename: str = "db-naive-llama.sqlite"
	):
		self.con = sqlite3.connect(filename)
		self.cur = self.con.cursor()
		self.create_table()

	def create_table(self):
		self.cur.execute(
			"""
			CREATE TABLE IF NOT EXISTS records (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				world_size INTEGER,
				batch_size INTEGER,
				input_len INTEGER,
				output_len INTEGER,
				prefill_time_usage REAL,
				decoding_time_usage REAL,
				tag VARCHAR
			)
			"""
		)
		self.con.commit()
	
	def _get_tag(
		self,
		model_path: str,
		mode: list[str]
	) -> str:
		return model_path + "," + str(mode)
	
	def check_if_record_exists(
		self,
		model_path: str,
		mode: list[str],
		world_size: int,
		batch_size: int,
		input_len: int,
		output_len: int,
		max_total_token_num: int
	) -> bool:
		tag = self._get_tag(model_path, mode)
		self.cur.execute(
			"""
			SELECT * FROM records WHERE world_size = ? AND batch_size = ? AND input_len = ? AND output_len = ? AND tag = ?
			""",
			(world_size, batch_size, input_len, output_len, tag)
		)
		return self.cur.fetchone() is not None
	
	def update_or_insert_record(
		self,
		model_path: str,
		mode: list[str],
		world_size: int,
		batch_size: int,
		input_len: int,
		output_len: int,
		prefill_time_usage: float,
		decoding_time_usage: float,
		max_total_token_num: int
	):
		tag = self._get_tag(model_path, mode)
		if self.check_if_record_exists(model_path, mode, world_size, batch_size, input_len, output_len, max_total_token_num):
			self.cur.execute(
				"""
				UPDATE records SET 
					prefill_time_usage = ?,
					decoding_time_usage = ?
					WHERE
						world_size = ? AND
						batch_size = ? AND
						input_len = ? AND
						output_len = ? AND
						tag = ?
				""",
				(prefill_time_usage, decoding_time_usage,
	 			world_size, batch_size, input_len, output_len,
				tag)
			)
		else:
			self.cur.execute(
				"""
				INSERT INTO records (world_size, batch_size, input_len, output_len, prefill_time_usage, decoding_time_usage, tag) VALUES (?, ?, ?, ?, ?, ?, ?)
				""",
				(world_size,
	 			batch_size, input_len, output_len,
				prefill_time_usage, decoding_time_usage,
				tag)
			)
		self.con.commit()
