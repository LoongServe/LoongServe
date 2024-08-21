import sqlite3
import math
import numpy as np
import csv

def read_data(
    sp_world_size: int,
    tp_world_size: int
) -> list:
    con = sqlite3.connect("db-identical-req.sqlite")
    cur = con.cursor()
    cur.execute(f"SELECT batch_size, input_len, avg_decoding_time_usage FROM records WHERE sp_world_size=? AND tp_world_size=?",
             (sp_world_size, tp_world_size))
    return [
        [float(row[0]), float(row[1]), float(row[2])]
        for row in cur.fetchall()
    ]

def evaluate(data, predict_func):
    batch_size = len(data)
    rel_errs = []
    print(f"{'bs':>2s}  {'ilen':>6s}  {'actual':>9s}  {'pred':>9s}  {'rel_err':>6s}")
    for i in range(batch_size):
        predicted = predict_func(data[i])
        rel_err = (predicted - data[i][2])/data[i][2]
        print(f"{data[i][0]:2.0f}  {data[i][1]:6.0f}  {data[i][2]:9.1f}  {predicted:9.1f}  {rel_err*100:6.2f}%")
        rel_errs.append(abs(rel_err))
    avg_rel_err = np.mean(rel_errs)
    avg_rel_err2 = np.mean([rel_err**2 for rel_err in rel_errs])**0.5
    print(f"avg_rel_err  = {avg_rel_err*100:.2f}%")
    print(f"avg_rel_err2 = {avg_rel_err2*100:.2f}%")

def approach_linalg(data):
    abc_arr = np.array([
        [row[0]*row[1], row[0]*row[1]**2, row[2]]
        for row in data
    ])
    batch_size = len(data)
    def weight(time_usage):
        return time_usage**0.5
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
    csv_file_path = "profiler_parameters.csv"
    with open(csv_file_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["sp_world_size", "tp_world_size", "batch_size", "A", "B", "C"])
        writer.writeheader()

        for sp_world_size, tp_world_size in [
            (1, 1),
            (2, 1), (1, 2),
            (4, 1), (2, 2), (1, 4),
            (8, 1), (4, 2), (2, 4), (1, 8)
        ]:
            print(f"---------------------------------------------------------------")
            print(f"Running for sp_world_size={sp_world_size}, tp_world_size={tp_world_size}")
            data = read_data(sp_world_size, tp_world_size)
            # approach_nn(data)
            coef = approach_linalg(data)
            writer.writerow({
                "sp_world_size": sp_world_size,
                "tp_world_size": tp_world_size,
                "batch_size": data[0][0],
                "A": coef[0],
                "B": coef[1],
                "C": coef[2]
            })
