# Test scripts for
# Strategic Heuristic Search with Batch Edits and Strategy Evaluation

from datamodel_esn import DataModel, ESN
from theta_regress import Theta, Regress
from hs_run        import HS_ESN
from hs_esn_plots  import HS_ESN_plots

import numpy as np
import matplotlib.pyplot as plt
import csv
import time

# ========================================================================
# HS model-running instance
#hs = HS_ESN('ESN Baselines',
hs = HS_ESN('ESN-Baselines',
            nodes_per_chan= 100,    # 100 --> D=800
            train_per_node= 100)    # 100 --> n_train=80000
start_time = time.perf_counter()
time1 = start_time

# Run ESN models
#for i in [0,1,3,4,6]:
for i in [2,5]:
    result, rtruth = hs.run_ESN(init_W=i)
    file_traj = hs.save(result, i)
    time2 = time.perf_counter()
    print(f'{(time2-time1)/60:.1f} minutes; T{i} saved to {file_traj=}')
    time1 = time2
file_true = hs.save(rtruth, -1)
print(f'True trajectory saved to {file_true=}')

# =============================================================================
# # Run HS models
# hs.name='HS vs ESN'
# 
# for i in [0, 1, 3, 6]:  # init W to I=original ESN, T_1=classic ESN, T_3=Ashesh
#     result, rtruth = hs.run_HS(init_W=i, max_iter=200, bat_size=3)
#     file_traj = hs.save(result, i)
#     time2 = time.perf_counter()
#     print(f'{(time2-time1)/60:.1f} minutes; T{i} saved to {file_traj=}')
#     time1 = time2
# =============================================================================

# ========================================================================
