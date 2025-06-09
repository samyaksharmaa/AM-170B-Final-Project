#!/usr/bin/env python
# coding: utf-8

# Hybrid ESN-SINDy HS and model-running functions
# AM 170B, Group 6
# version 26may2025
# Edwin Hutchins - overall organization
# Sidd Singh - heuristic search functions
# Brayan Vaca - plotting and statistical summaries

# %%

from datamodel_esn import DataModel, ESN
from theta_regress import Theta, Regress

import numpy as np
import time
import matplotlib.pyplot as plt
import csv

# ----------------------------------------------------------------------------
#   Global utilities / helper functions
# ----------------------------------------------------------------------------
def forecast_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#??? CAUTION: abbrev must be kept in synch with HS_ESN_plots.T_labels(abbrev)
abbrev = ['T_0', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'na', 'na',
          'HS0', 'HS1', 'HS2']

# HS -- Heuristic Search of polynomial term space for Theta.W. Includes:
#    *  Heuristics to help guide search for optimal regression terms.
#    *  Iteration functions to cycle through multiple search strategies.
#    *  High-level management of batch and iterative strategies.
#    *  Automated running of bundles of models, for stats and plotting

# %%
# ----------------------------------------------------------------------------
# HS_ESN defines a runtime model and provides high-level HS and run functions
# Scaling arguments maintain ratios across different overall sizes:
#   nodes_per_channel scales reservoir size to match number of input channels
#   train_per_node scales n_train to reservoir size
# ----------------------------------------------------------------------------
class HS_ESN():
    def __init__(self, name='HS_ESN',
                 nodes_per_chan=  100,   # 100 --> D=800
                 train_per_node=  100,   # 100 --> n_train=80k
                 n_predict     = 1000,   # standard for plots is 1000
                 msg_list=None):
        '''Hybrid ESN with Heuristic Search: wrapper for model build and run.
        Set msg_list=False to disable logging.'''
        self.name   = name                      # for documentation, plots
        self.NPC    = nodes_per_chan
        self.TPN    = train_per_node
        self.n_predict = n_predict
        self.dm = dm = DataModel(name)              # Read test data
        self.C      = self.dm.dim()[0]              # n input channels
        self.D      = self.C * nodes_per_chan
        self.n_train= self.D * train_per_node
        self.data   = dm.data(0, self.n_train)          # ptr to training data
        self.truth  = dm.data(self.n_train, n_predict)  # ptr to test data
        self.run_name = f'{name}: res_size={self.D}, n_train={self.n_train}'
        self.esn   = ESN('Basic ESN', dm, res_size=self.D)      # layer 1
        self.reg   = Regress('Basic ESN', dm)                   # layer 3
        self.theta = Theta(self.esn, self.reg)                  # layer 2
        self.E     = self.D                         # out to reg; may change
        self.log   = []                             # place to store HS details
        self.bsize = []                             # batch size by strategy
# =============================================================================
#         self.save  = False                          # True to save output files
# =============================================================================

    # ------------------------------------------------------------------------
    #  Helper functions for the run_*() methods:
    #    set_W()     model definition, for internal and external use
    #    _get_r()    used by Run_* methods for setup
    #    _get_Wout() used by Run_* methods to train Wout
    #    _predict()  used by Run_* to predict a trajectory
    #    _truth()    get true values of predicted trajectory
    #  (analysis and reporting functions are further down, in last section)
    # ------------------------------------------------------------------------

    def set_W(self, init_W, rows=None):
        '''Initialize theta.W to specific ESN, HS, or SINDy version.
        For HS models, init_W is +10; for SINDy (when ready) it is +20.'''
        theta = self.theta
        if init_W < 10:
            theta.set_W_Ti(T_i=init_W)  # 0=I, 1=classic, 2-3=Ashesh, 6=random
        elif init_W < 20:
            theta.set_W_HS(diag=init_W-10, n_out=rows)  # diagonal, number rows
        else:
            theta.set_W_SINDy(degree=init_W-20)         # only supports 1 or 2
        return

    def _get_r(self, init_W, seed=0):
        '''Set up a single model. Default Theta.W = I (no Theta function).
        Initialize W via init_W, a tuple of (ESN,HS,SINDy) versions'''
        n_train = self.n_train                  # training time steps
        dm  = self.dm
        esn = self.esn
        esn.generate(seed=seed)
        self.set_W(init_W)                      # set_W can also be re-called
        r = esn.train(self.data, n_train)
        return r

    def _get_Wout(self, r):
        '''Train, save output. Saves Wout internally.'''
        X    = self.theta.run(r)                # uses current theta.W values
        Wout = self.reg.train(X, self.data)
        self.Wout = Wout
        return Wout

    def _predict(self, r, Wout=None):
        if Wout is None: Wout = self.Wout
        tP = self.n_predict
        output = np.zeros((self.C, tP))         # for model performance plots
        ## states = np.zeros((self.D, tP))      # diagnostic use only
        if   r is None:      r=self.esn.predict(None)  # final training state
        elif len(r.shape)>1: r = r[:,-1]
        for t in range(tP):
            X   = self.theta.run(r)             # regression input vector
            out = self.reg.predict(X)
            output[:,t] = out                   # predicted trajectory
            r = self.esn.predict(r, out)
            ## states[:,t] = r                  # diagnostic use only
        # Save predicted values for plots
        self.output = output  #??? DO WE EVER USE THIS ########################
        return output

    # ------------------------------------------------------------------------
    # Heuristic search functions - from Sidd's strategic_hs_pipeline.py
    # ------------------------------------------------------------------------

    def apply_strategy(self, strategy_id):
        rows_affected = self.log = []
        batch_size    = self.bsize
        esn   = self.esn
        theta = self.theta
    
        if strategy_id == 0:  # Add
            for _ in range(batch_size[0]):
                indices = list(np.random.choice(esn.D, 3, replace=False))
                exponents = [1, 1, 1]
                row_idx = theta.add(indices, exponents)
                idxs = [int(indices[i]) for i in range(len(indices))]
                rows_affected.append(("add", int(row_idx), idxs, exponents))
    
        elif strategy_id == 1:  # Duplicate
            #??? 5/26/2025 Fixed .duplicate(); break ties by adding 1 to diag
            for _ in range(batch_size[1]):
                row = np.random.randint(0, theta.E)
                new_row = theta.duplicate(row)
                rows_affected.append(("duplicate", int(new_row), None, None))

        elif strategy_id == 2:  # Replace
            for _ in range(batch_size[2]):
                row = np.random.randint(0, theta.E)
                indices = list(np.random.choice(esn.D, 2, replace=False))
                exponents = [2, 1]
                theta.replace(row, indices, exponents)
                idxs = [int(indices[i]) for i in range(len(indices))]
                rows_affected.append(("replace", int(row), idxs, exponents))

        if strategy_id == 3:  # Use Regress.cross_cov() to select rows to add
            self.new_terms(min_rel=0.0001, how_many=batch_size[3])

        elif strategy_id == 4:  # Duplicate using Theta.tune_W()
            # Extends strategy 1: Run tune_W() to adjust exponents
            self.dup_terms(min_rel=1.0, how_many=batch_size[4])

        return rows_affected
        # end apply_strategy()

    # HS main strategic operations center; manages iterations
    # Renamed from search_theta_with_strategies()
    # Returns best_mse, stategy_log
    def search_main(self, r, max_iters=20, batch_size=5, verbose=True,
                    log_path="strategy_log.csv"):
        '''Heuristic search core method. Explores term space by cycling 
        through different strategies for morphing theta.W. batch_size controls
        how many mutations generated on each iteration; may be single int or
        a list of values for each strategy. r is esn training output.'''
        data  = self.data               # training data
        truth = self.truth              # prediction target values
        n_train=self.n_train
        dm    = self.dm
        esn   = self.esn
        reg   = self.reg
        theta = self.theta
        self.bsize = [batch_size]*5 if isinstance(batch_size,int) else batch_size
        batch_size = self.bsize

        best_W = theta.W.copy()
        best_Y = theta.run(r)
        best_Wout = reg.train(best_Y, data)
        
        # start predicted trajectory with esn final training state r[-1]
        best_pred = self._predict(r, best_Wout)
        best_mse  = forecast_mse(truth, best_pred)
        # =======================================================================
        strategy_log = []
    
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Strategy_ID", "MSE_Before",
                             "MSE_After", "Improvement", "Minutes", "Edits"])
            start_time = time.perf_counter()
            time1 = start_time

        for i in range(max_iters):
            strategy_id = i % 5  # Rotate through strategies
            # suppress a strategy if no longer effective
            if batch_size[strategy_id] <= 0:
                continue
            theta.W = best_W.copy()  # Start fresh from best so far
            edits = self.apply_strategy(strategy_id)
    
            try:
                Y = theta.run(r)
                Wout = reg.train(Y, data)
                pred = self._predict(r, Wout)
                mse  = forecast_mse(truth, pred)
            except Exception as e:
                if verbose:
                    print(f"Iteration {i} failed: {e}")
                continue
            time2 = time.perf_counter()
            minutes = (time2-time1)/60
            time1 = time2
            
            improvement = best_mse - mse
            # Log each iteration
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i, strategy_id, best_mse, mse,
                                 improvement, minutes, edits])
            if mse < best_mse:
                best_mse = mse
                best_W = theta.W.copy()
                self.bsize[strategy_id] += 1    # expand use if success
            else:
                self.bsize[strategy_id] -= 1    # decrease use if failure

            if i == 0:
                print(f"Initial MSE: {best_mse:.6f}")
            if verbose:
                print(f"Iter {i}: Strategy {strategy_id}, MSE={mse:.6f},",
                      f"{minutes=:.2f}, {improvement=:.6f}")
    
            strategy_log.append((i, strategy_id, improvement, minutes))
            
            # solution has converged if all strategies no longer working
            if sum(batch_size) <= 0:
                break
        theta.W = best_W
        return best_mse, strategy_log
        # end search_main()
    
    # ========================================================================
    # Run the model - Start with core run() method; duplicate as needed with
    #   new name here; or in a separate file, either adding or replacing:
    # def: my_run_function(): return('This is a new HS_ESN method!')
    # HS_ESN.run_new = my_run_function  # adds a new method
    # HS_ESN.run = my_run_function      # replaces the existing method
    # ========================================================================
    # Simplest model = static Theta.W
    def run_ESN(self, init_W, save=None):
        '''Set up, run, save a single basic model with fixed theta.W
        init_W specifies Theta.W initialization: <10=Ti(ESN), <20=HS(diag)'''
        r     = self._get_r(init_W)
        Wout  = self._get_Wout(r)               # calls reg.train()
        traj  = self._predict(r, Wout)
        return (self.package(traj, init_W),
                self.package(self.truth, -1))

    # Basic heuristic search (single model)
    def run_HS(self, init_W=11, max_iter=200, bat_size=5, name=None,
               rows=None, save=None):
        '''Set up, run, save an HS model where theta.W is learned.
        init_W can be an ESN model(<10) or diagonal matrix (10-12).
        If rows is specified, W will be truncated or extended with zeros.
        Default W initialization is the identity matrix.'''
        r    = self._get_r(init_W, rows)        # run ESN, initialize Theta.W
        Wout = self._get_Wout(r)                # initial solution
        # Heuristic search calls theta.run() and reg.train() repeatedly
        # At end, theta.W is optimized, and prediction plots are as usual 
        # search_main() renamed from search_theta_with_strategies()
        fname = self.name if name is None else name
        filename = f'{fname}_D{self.D:04d}_T{self.n_train:06d}_M{init_W:02d}.csv'
        best_mse, hs_log = self.search_main(r, max_iter, bat_size,
                                            # verbose=False,
                                            log_path=filename)
        print(f"Final Best MSE: {best_mse:.6f} on iteration {len(hs_log)}")

        Wout = self._get_Wout(r)
        traj = self._predict(r, Wout)
        return (self.package(traj, init_W),
                self.package(self.truth, -1))
        
    def run_many(self, init_W, n_runs=1, report_each=True, save=None):
        '''Set up a single model, run multiple times, summarize, plot.'''
        core_name = self.name                   # batch number appended in loop
        results = []
        for i in range(n_runs):
            self.esn.generate(seed=i)           # re-randomize ESN (A, Win)
            r    = self._get_r(init_W)
            Wout = self._get_Wout(r)
            traj = self._predict(r, Wout)
            self.name = f'{core_name}_{i:02d}'
            results.append((self.package(traj, init_W),
                            self.package(self.truth, -1)))
        # Restore model name before returning
        self.name = core_name
        return results          # list of tuples; each run has its own truth
        # end run_many()

    # Loop model variations; see HS_ESN.set_W() for argument details
    # set_W will use the first valid argument it finds: ESN, HS or SINDy;
    # depending on which is found first, branches to that model
    def run_set(self, models, bat_size=5, max_iter=20, rows=None, save=False):
        '''Set up a series of models, run, plot comparisons.
        models is a list of tuples: (model{0=ESN,1=HS} and version{0-12}.
        rows is only used for model 1 (HS): None=square, >0=rectangular.'''
# =============================================================================
#         if save is noFt None: self.save = save
# =============================================================================
        #    example = [
        #        (0, 1),        # T_1 = classic ESN (even nodes squared)
        #        (0, 3),        # T_3 = Ashesh ESN (even=product of neighbors)
        #        (0, 6),        # T_6 = ESN with random quadratic terms
        #        (1,10),        # HS initialized to all zeros
        #        (1,11),        # HS initialized to I (identity matrix)
        #        (1, 3)]        # HS initialized to T_3 (Ashesh model)
        mod_abb   = ['ESN', 'HS', 'SINDy']
        core_name = self.name   # batch number appended in loop
        results   = []

        rtruth = self.package(self.truth, -1)    # model -1 = truth

        for i in range(len(models)):
            mod_typ, mod_ver = models[i]
            self.name = f'{core_name}_{mod_abb[model]}{version}'
            if  mod_typ==0:
                traj = run_ESN(mod_ver, save=save)
            elif mod_typ==1:
                traj, _,_,_ = run_HS(m0d_ver, bat_size, max_iter,
                                     rows=rows, save=save)
            else:
                print(f'run_set(): {mod_typ=} ignored: not yet implemented')
                break
            result = self.package(traj, mod_ver)
            results.append(result)
        self.name = core_name                   # restore model name
        return (result, rtruth)
        # end run_set()

# %%
# ========================================================================
# Performance analysis and plotting functions
# See file hs_esn_plotting_with_save2.py  ( --> hs_esn_plot_save.py #???)
# ========================================================================

    # -------------------------------------------------------------------------
    # Heuristic methods that provide hints for selecting new polynomial+ terms
    #   or directly manipulate the Theta.W coefficient matrix.
    # -------------------------------------------------------------------------

    # Compute and sort relevance for all pairings of current terms
    def new_terms(self, min_rel=1.0, how_many=None):
        '''Add new theta.W terms selected from all pairs of existing terms.
        Based on relevance scores from Regress.cross_rel().
        Returns list of added rows, array of relevance scores sorted descending.
        Relevance = new term std deviation * magnitude of regression coefft.'''
        theta = self.theta
        reg   = self.reg
        try:    
            E = reg.E
        except:
            print('HS.new_terms() aborted: call Regress.train(), set Theta.reg')
            return None
        # Current calc is very slow; sample training data to speed it up
        step = int(np.ceil(self.n_train / np.random.randint(2000,2100)))
        idx, rel = reg.cross_rel(step=step)     # W indices, descending relev.
        batch_size = int(E*(E+1)/2)
        if how_many is not None and how_many < batch_size:
            batch_size = how_many
        rows = []
        for i in range(batch_size):
            new_row  = (theta.W[idx[0,i],:] + theta.W[idx[1,i],:]).real
            indices  = np.nonzero(new_row)[0].tolist()
            exponents= new_row[indices].tolist()
            if rel[i] < min_rel:
                break
            row = theta.add(indices, exponents)
            rows.append(row)
            self._log("add", row, indices, exponents)
        return rows, rel[:len(rows)]
    
    # Adjust theta exponents (theta.W)
    def dup_terms(self, min_rel=1.0, how_many=None, 
                  max_iter=2, l2_weight=0.00001, lrlr=0.001):
        '''Optimize polynomial exponents in Theta.W for a fixed Regress.Wout.
        Does not alter structure of W, only modifies existing non-zero cells.
        Assumes caller has called reg.train().
        Returns a list of suggested alterations [(weight,row,type), ...]'''
        reg   = self.reg
        theta = self.theta
        
        # Get loss gradients from regression and l2 norm
        _,_,_,dLdP,_ = reg.grad()       # regression dLoss/dX
        # Learn theta.W using back-prop of loss gradient, within fixed Wout's
        for _ in range(max_iter):
            theta.tune_W(dLdP, l2weight=l2_weight, lrlr=lrlr)
        # select best row scores = exponents between 2 integers
        idx, score = theta.find_splits(rel=False)
        ### rel=False SUPPRESSES RELEVANCE IN SCORE --> BUG: TRASHES MSE #???
        for i in range(how_many):
            row = idx[i]
            if score[idx[i]] >= min_rel:
                new_row = theta.duplicate(row)
                w = theta.W.real[new_row,:]
                indices = np.nonzero(theta.W[row,:])[0]
                exponts = w[indices]
                self._log("duplicate", new_row, indices, exponts)
            # else there are no non-integers in remaining rows
        # end dup_terms()
        
    # -------------------------------------------------------------------------
    # Internal utilities
    # -------------------------------------------------------------------------
    
    # Log HS messages to track details of each iteration
    def _log(self, message, row, indices, exponents):
        if self.log:
            idx = [int(indices[i]) for i in range(len(indices))]
            self.log.append((message, int(row), idx, exponents))

    def get_filename(self, model, name=None):
        if type(model) != int: print(f'write(): {model=}')
        mod = 'truth' if model == -1 else f'{model:02d}'  # abbrev[model]
        name = self.name if name is None else name
        fname = name.replace(' ', '-')
        filename = f'{fname}_D{self.D:04d}_T{self.n_train:06d}_M{mod}.npz'
        return filename

    def package(self, traj, model):
        '''Combine results into a dictionary in same format as read()'''
        result = {'name':self.name, 'D':self.D, 'E':self.E, 'T':self.n_train,
                  'model':model, 'traj': traj}
        return result

    def save(self, result, model):
        '''Save trajectory to disk, including parameters; input a dictionary'''
        filename = self.get_filename(model)
        np.savez_compressed(filename, **result)
        return filename

    # Old way, depracated; use save(package(traj, model), model)
    def write(self, result, model):
        '''Save a trajectory to disk, including parameters; traj is array'''
        if type(model) != int: print(f'write(): bad {model=}') #??? ############
        filename = self.get_filename(model)
        name = np.array(self.name)
        a_D = np.array(self.D)
        a_E = np.array(self.E)
        a_T = np.array(self.n_train)
        a_model = np.array(model)       # index; ESN 0-7, HS 10-12, truth -1
        np.savez_compressed(filename, name=name,
                            D=a_D, E=a_E, T=a_T, model=a_model, traj=traj)
        return filename

    def read(self, filename):
        '''Read a trajectory or set of trajectories from disk'''
        print(f'Reading results from file {filename}')
        name = filename[:filename.find('_D')].replace('-', ' ')
        pkg = np.load(filename)
        result = {
            'name':  name,
            'D':     pkg['D'].item(),
            'E':     pkg['E'].item(),
            'T':     pkg['T'].item(),
            'model': pkg['model'].item(),
            'traj':  pkg['traj']
            }
        return result

    def read_model(self, model, name=None):
        filename = self.get_filename(model, name)
        return self.read(filename)
