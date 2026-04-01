import time
import math
from pathlib import Path

import numpy as np
from qcodes.dataset import (
    Measurement,
    initialise_or_create_database_at,
    load_or_create_experiment,
)

DB_PATH = Path(r"c:\Users\francescozatel\Code\KitaevPapers\qubit_pmm\raw_data\U302A_dev3_21.db")

initialise_or_create_database_at(str(DB_PATH))
exp = load_or_create_experiment(
    experiment_name="live_plottr_test_25x25",
    sample_name="live_dummy_sample",
)

meas = Measurement(exp=exp)
meas.register_custom_parameter("live_x", unit="a.u.")
meas.register_custom_parameter("live_y", unit="a.u.")
meas.register_custom_parameter("live_signal", unit="V", setpoints=("live_x", "live_y"))

xs = np.linspace(-1.0, 1.0, 25)
ys = np.linspace(-1.0, 1.0, 25)

with meas.run() as datasaver:
    run_id = datasaver.run_id
    print(f"LIVE_TEST_RUN_ID={run_id}", flush=True)

    total = len(xs) * len(ys)
    count = 0
    for x in xs:
        for y in ys:
            # Smooth 2D pattern with mild drift to make live updates visible.
            z = math.sin(2.0 * math.pi * 2.5 * x) * math.cos(2.0 * math.pi * 3.0 * y) + 0.05 * x
            datasaver.add_result(("live_x", float(x)), ("live_y", float(y)), ("live_signal", float(z)))
            count += 1
            if count % 25 == 0:
                print(f"PROGRESS={count}/{total}", flush=True)
            time.sleep(0.5)

print("LIVE_TEST_DONE=1", flush=True)
