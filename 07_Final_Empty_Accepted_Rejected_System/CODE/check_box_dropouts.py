import csv, sys, glob, os
from collections import defaultdict

folder = r"C:\Do_Not_Delete_PLC\Conference\07_Final_Empty_Accepted_Rejected_System\RESULTS\50_RFID1_Ground_Truth_Log"
path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(os.path.join(folder, "frames_*.csv")), key=os.path.getmtime)
print("file:", path)

frames = defaultdict(list)
times = {}
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        n = int(r["frame"])
        frames[n].append((int(r["track"]), float(r["cx"]), float(r["cy"]), int(r["missed"])))
        times[n] = float(r["t_rel_sec"])

order = sorted(frames)
counts = [len(frames[n]) for n in order]
print("frames:", len(order), " box count min/max:", min(counts), max(counts))
usual = max(set(counts), key=counts.count)
print("most common count:", usual)

start = None
for n, c in zip(order, counts):
    if c < usual and start is None:
        start = n
    if c >= usual and start is not None:
        print(f"count below {usual}: frames {start}-{n-1}  {times[start]:.2f}s-{times[n-1]:.2f}s  length {times[n-1]-times[start]:.2f}s")
        start = None

held = sum(1 for n in order for b in frames[n] if b[3] > 0)
print("box-frames drawn as HELD (missed>0):", held)