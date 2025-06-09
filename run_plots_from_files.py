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

# HS_ESN instance for access to file-reading and plotting functions
#hs = HS_ESN('Stub', nodes_per_chan=100)         # D800 files
hs = HS_ESN('Stub', nodes_per_chan=300)         # D2400 files

# File selection method 1 - full filename
# =============================================================================
# filename = 'T6-Test-Runs_D0320_T032000_MT_6.npz'
# result = hs2.read(filename)
# =============================================================================

# File selection method 2 - specify model index and core name
folder=''
folder='../data_upload/'
true = hs.read_model(-1, name=folder+'ESN Baselines')

esn0 = hs.read_model( 0, name=folder+'ESN Baselines')
esn1 = hs.read_model( 1, name=folder+'ESN Baselines')
#esn2 = hs.read_model( 2, name=folder+'ESN Baselines')
esn3 = hs.read_model( 3, name=folder+'ESN Baselines')
#esn4 = hs.read_model( 4, name=folder+'ESN Baselines')
#esn5 = hs.read_model( 5, name=folder+'ESN Baselines')
esn6 = hs.read_model( 6, name=folder+'ESN Baselines')

hs_0 = hs.read_model( 0, name=folder+'HS vs ESN')
hs_1 = hs.read_model( 1, name=folder+'HS vs ESN')
hs_3 = hs.read_model( 3, name=folder+'HS vs ESN')
hs_6 = hs.read_model( 6, name=folder+'HS vs ESN')

# File selection method 3 - specify model index; pulls name from HS_ESN
# =============================================================================
# hs2    = HS_ESN('T6 Test Runs', nodes_per_chan=40)
# rtruth = hs2.read_model(-1)
# =============================================================================

# ----------------------------------------------------------------------------
#                                       Plots
# ----------------------------------------------------------------------------

hp = HS_ESN_plots(hs)

# Explore ESN
#results = [esn0, esn1, esn2, esn3, esn4, esn5, esn6]
results = [esn0, esn1, esn3, esn6]
hp.report(results, true)

# Goal 1: beat T1 (classic ESN) via HS. Plot: compare HS-T0 to T0 and T1.
results = [esn0, esn1, hs_0]
hp.report(results, true)

# Goal 2,3: improve, beat T3 (Chat 2020). Plot: compare HS-T3 to T1 and T3.
results = [esn1, esn3, hs_1]
hp.report(results, true)

# Goal 4,5: beat T3 in less time, improve on T6 using generic HS.
# Plot: compare HS-T6 to T3, T6.
results = [esn1, esn3, hs_6]
hp.report(results, true)
