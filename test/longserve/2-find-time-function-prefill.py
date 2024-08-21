"""
Find the prefill-stage time function

The function looks like:

Time consumption = A + B * sum(l_i) + C * sum(l_i^2), where l_i is the length
of the i-th request
"""
import sqlite3
import math
import numpy as np
import csv
import argparse

def read_data(
    db_path: str,
    sp_world_size: int,
    tp_world_size: int
) -> list:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(f"SELECT batch_size, input_len, avg_prefill_time_usage FROM records WHERE sp_world_size=? AND tp_world_size=? ORDER BY input_len ASC, batch_size ASC",
             (sp_world_size, tp_world_size))
    return [
        [float(row[0]), float(row[1]), float(row[2])]
        for row in cur.fetchall()
    ]

def geo_mean(iterable):
    return np.exp(np.log(iterable).mean())

def evaluate(data, predict_func):
    batch_size = len(data)
    rel_errs = []
    print(f"{'bs':>3s}  {'ilen':>6s}  {'actual':>9s}  {'pred':>9s}  {'rel_err':>6s}")
    for i in range(batch_size):
        predicted = predict_func(data[i])
        rel_err = (predicted - data[i][2])/data[i][2]
        print(f"{data[i][0]:3.0f}  {data[i][1]:6.0f}  {data[i][2]:9.1f}  {predicted:9.1f}  {rel_err*100:6.2f}%")
        rel_errs.append(abs(rel_err))
    avg_rel_err = np.mean(rel_errs)
    geomean_rel_err = geo_mean(rel_errs)
    print(f"avg_rel_err  = {avg_rel_err*100:.2f}%")
    print(f"geomean_rel_err = {geomean_rel_err*100:.2f}%")

def approach_linalg(data):
    abc_arr = np.array([
        [row[0]*row[1], row[0]*row[1]**2, row[2]]
        for row in data
    ])
    batch_size = len(data)
    def weight(time_usage):
        return time_usage**0.1
    A = np.array([
        [1/c*weight(c), a/c*weight(c), b/c*weight(c)]
        for a, b, c in abc_arr
    ])
    b = np.array([
        weight(c)
        for a, b, c in abc_arr
    ])
    coef, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print(f"Coefs: {coef}")

    def predict(row):
        return coef[0] + coef[1]*row[0]*row[1] + coef[2]*row[0]*row[1]**2
    evaluate(data, predict)
    return coef

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-db", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--enable-multi-node", action="store_true")
    args = parser.parse_args()
    csv_file_path = args.output_csv
    with open(csv_file_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["sp_world_size", "tp_world_size", "batch_size", "A", "B", "C"])
        writer.writeheader()

        for sp_world_size, tp_world_size in [
            (sp_world_size, tp_world_size)
            for tp_world_size in [1,2,4,8]
            for sp_world_size in [1,2,3,4,5,6,7,8]
            if sp_world_size*tp_world_size <= 8 or (tp_world_size == 2 if args.enable_multi_node else False)
        ]:
            print(f"---------------------------------------------------------------")
            print(f"Running for sp_world_size={sp_world_size}, tp_world_size={tp_world_size}")
            data = read_data(args.profile_db, sp_world_size, tp_world_size)
            coef = approach_linalg(data)
            writer.writerow({
                "sp_world_size": sp_world_size,
                "tp_world_size": tp_world_size,
                "batch_size": data[0][0],
                "A": coef[0],
                "B": coef[1],
                "C": coef[2]
            })
