import sqlite3
import copy

from lib.structs import *

class RecordManager:
    """
    To speed up our experiment, we store previous experiment results in a database.
    """
    def __init__(
        self,
        filename
    ):
        self.con = sqlite3.connect(filename)
        self.con.row_factory = sqlite3.Row
        self.cur = self.con.cursor()
        self._create_table()

    def _create_table(self):
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sp_world_size INTEGER,
                tp_world_size INTEGER,
                batch_size INTEGER,
                input_len INTEGER,
                output_len INTEGER,
                num_sp_master INTEGER,
                need_context_migration BOOLEAN,
                num_decoding_stage_migration INTEGER,
                avg_prefill_time_usage REAL,
                avg_decoding_time_usage REAL,
                prefill_time_stddev REAL,
                decoding_time_stddev REAL,
                tag VARCHAR
            )
            """
        )
        self.con.commit()
    
    def _get_tag(
        self,
        worker_param: WorkerParam,
    ) -> str:
        return worker_param.model_dir + "," + str(worker_param.mode)
    
    def query_record(
        self,
        worker_param: WorkerParam,
        input_param: InputParam
    ) -> sqlite3.Row:
        """
        Query the record from the database. If the record does not exist, return None
        """
        tag = self._get_tag(worker_param)
        self.cur.execute(
            """
            SELECT * FROM records
            WHERE sp_world_size = ? AND tp_world_size = ? AND
                batch_size = ? AND input_len = ? AND output_len = ? AND num_sp_master = ? AND
                need_context_migration = ? AND num_decoding_stage_migration = ? AND
                tag = ?
            """,
            (worker_param.sp_world_size, worker_param.tp_world_size,
             input_param.batch_size, input_param.input_len, input_param.output_len, input_param.num_sp_master,
             input_param.need_context_migration, input_param.num_decoding_stage_migration,
             tag)
        )
        return self.cur.fetchone()
    
    def update_or_insert_record(
        self,
        worker_param: WorkerParam,
        input_param: InputParam,
        avg_prefill_time_usage: float,
        avg_decoding_time_usage: float,
        prefill_time_stddev: float,
        decoding_time_stddev: float
    ):
        """
        Update or insert a new record
        """
        tag = self._get_tag(worker_param)
        if self.query_record(worker_param, input_param) != None:
            self.cur.execute(
                """
                UPDATE records SET 
                    avg_prefill_time_usage = ?,
                    avg_decoding_time_usage = ?,
                    prefill_time_stddev = ?,
                    decoding_time_stddev = ?
                    WHERE
                        sp_world_size = ? AND
                        tp_world_size = ? AND
                        batch_size = ? AND
                        input_len = ? AND
                        output_len = ? AND
                        num_sp_master = ? AND
                        need_context_migration = ? AND
                        num_decoding_stage_migration = ? AND
                        tag = ?
                """,
                (avg_prefill_time_usage, avg_decoding_time_usage, prefill_time_stddev, decoding_time_stddev,
                 worker_param.sp_world_size, worker_param.tp_world_size,
                input_param.batch_size, input_param.input_len, input_param.output_len, input_param.num_sp_master,
                input_param.need_context_migration, input_param.num_decoding_stage_migration,
                tag)
            )
        else:
            self.cur.execute(
                """
                INSERT INTO records (sp_world_size, tp_world_size, batch_size, input_len, output_len, num_sp_master, need_context_migration, num_decoding_stage_migration, avg_prefill_time_usage, avg_decoding_time_usage, prefill_time_stddev, decoding_time_stddev, tag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (worker_param.sp_world_size, worker_param.tp_world_size,
                 input_param.batch_size, input_param.input_len, input_param.output_len, input_param.num_sp_master,
                 input_param.need_context_migration, input_param.num_decoding_stage_migration,
                avg_prefill_time_usage, avg_decoding_time_usage, prefill_time_stddev, decoding_time_stddev,
                tag)
            )
        self.con.commit()
