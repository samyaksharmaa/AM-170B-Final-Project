#!/usr/bin/env python
# coding: utf-8

# Hybrid ESN-SINDy model core functions
# AM 170B, Group 6
# Derived from Chattopadhyay 2020 (ESN.py).
# Refactored and to facilitate experimentation and extension. Major changes:
# (1) Group training and prediction code into 3 layers with base classes
#   ESN = echo state network, AKA reservoir - unique to ESN
#   Theta = ESN nonlinear transformation functions, SINDy theta function layer
#   Regress = regression layer
# (2) Plan: Extend or replace classes to build hybrid variations
# (3) Goals:
#   Improve prediction accuracy using the Chat 2020 paper as baseline
#   Reduce time to train while maintaining improved accuracy
#   Identify patterns in model dynamics, in the spirit of SINDy
#   Clarify ESN internal dynamics, surface topics for further development

# %%
import numpy as np
import scipy.stats  as st
import scipy.sparse as sparse
# from scipy.sparse import linalg
import pandas as pd  ### used once, for read_csv in DataModel: replace with np
import time
import matplotlib.pyplot as plt
# import logging

# ----- Class definitions -----
# %%
# DataModel -- global parameters, performance summary
#??? DEV NOTE: drop name for all classes except main (ie, HS_ESN)
class DataModel:
    def __init__(self, name='Chat-2020',
                 filename='3tier_lorenz_v3.csv', max_rows=None):
        self.name   = name      # descriptive label to help organize reporting
        self.file   = filename  # data source
        self._data  = None      # created in getx_data()
        self.report = 0         # print messages up to this level of loops
        self.read(max_rows)     # data may be reloaded later to get more rows

    def read(self, max_rows=None):
        '''generate or read data, save in internal array'''
        filename = self.file
        if self.report:
            print(f'{self.name}.read() reading file {filename}...')
        if self._data is None or max_rows != self.max_rows:
            dataf = pd.read_csv(filename, header=None, nrows=max_rows)
            self._data = np.transpose(np.array(dataf))
        # else we already have data
        
    def data(self, start=0, number=None):
        '''return the requested data slice; all records if no number'''
        assert self._data is not None, 'Call DataModel.read() before .data()'
        if number is None:
            return self._data[:, start:]
        return self._data[:, start:(start+number)]

    def dim(self, index=None):
        '''return dimensions of stored data array'''
        d = [0,0] if self._data is None else np.shape(self._data)
        return d if index is None else d[index]
    
    def show(self, level=0):
        '''print status update and set reporting level'''
        d = self.dim()
        print(f'DataModel {self.name}: {d[0]} input channels, {d[1]} time steps')
        self.report = level
# %%
# ESN (layer 1) -- implement the reservoir, provide core ESN functionality
class ESN:
    def __init__(self, name, dmodel, res_size=5000, radius=0.1, degree=3, sigma=0.5):
                 ##train_length=500000, predict_length=10000):
        self.name    = name             # for documentation
        self.dname   = dmodel.name      # for documentation
        chan         = dmodel.dim(0)    # length of input data vector (local var)
        # truncate reservoir size to a multiple of input channels
        self.D       = int(np.floor(res_size/chan) * chan)  # actual reservoir size
        self.n_chan  = chan             # input channels (rows); was num_inputs
        self.radius  = radius           # spectral radius of A
        self.degree  = degree           # average node degree (# links) in A
        self.sigma   = sigma            # input weight range: -sigma < Win <= sigma
        self.A       = None             # A and Win will be created by generate()
        self.Win     = None
        self.output  = None             # output data will be created by train()
        self.r_pred  = None             # current state for trajectory prediction
        self.report  = 0                # print messages up to this level of loops

    # ----- Callable methods -----

    # Generate reservoir (Win, A); state (r) created later when we read input data
    # Can be called repeatedly with different seeds to re-randomize A and Win each time
    def generate(self, seed=0, report=False):
        self._create_A(seed)            # A = ESN adjacency matrix; paper: 5Kx5K
        self._create_Win()              # Win = fixed random input weights

    # Create entire training output of state vectors r(t) = node weights at each time t
    # Each column of .output will hold r(t+1) = tanh(A x r(t) + Win x input(t))
    # Originally part of reservoir_layer()
    # 'data' is 2D array; each column is vector of channel states at one time step
    def train(self, data, nT):
        '''Update reservoir state from input data n times; save and return pointer.'''
        C    = self.n_chan
        D    = self.D
        size = np.shape(data)
        assert size[0] >= C,  f'data has {size[0]} rows < num channels {C=}'
        assert size[1] >= nT, f'data has {size[1]} columns < train size {nT=}'
        if self.A is None:
            self.generate()             # create A and Win from default seed
        states = np.zeros((D, nT))      # paper: 5K x 500K
        for i in range(nT-1):
            states[:, i+1] = np.tanh(np.dot(self.A, states[:, i]) +
                                     np.dot(self.Win, data[:, i]) )
        self.output = states
        return self.output              # returns pointer for easy access

    # Return next step in a predicted trajectory, for theta layer input
    # 'r' is Chat 2020 notation (state(t) in paper, x[i] in code)
    # 'chan' is the current system data vector: channel states at one time step
    def predict(self, r_in=None, chan=None):
        '''Predict next time step; if no input, return last training state'''
        assert self.output is not None, 'No output data: call train(data) first'
        if r_in is None:
            x0 = self.output[:,-1]             # start preds with last train
            return np.squeeze(np.asarray(x0))
        x1 = np.tanh(np.dot(self.A,   r_in) +
                     np.dot(self.Win, chan))    # ESN + weighted inputs
        r_pred = np.squeeze(np.asarray(x1))     # new state vector
        return r_pred

    # Display parameters
    def show(self):
        print(f'ESN reservoir {self.name}: data model={self.dname}, D={self.D},',
              f'n_chan={self.n_chan}, radius={self.radius}, degree={self.degree},',
              f'sigma={self.sigma}')

    # ----- Internal methods -----

    # Create matrix 'A' = adjacency matrix weighted by normalized reservoir flows
    # Originally generate_reservoir()
    def _create_A(self, seed):                  # paper: size=5000; radius=0.1; degree=3
        np.random.seed(seed=seed)
        D = self.D
        sparsity = self.degree/float(D);        # paper: 3/625 = .00512
        A = sparse.rand(D, D, density=sparsity).todense()
        # uniform random in [0,1); paper: in .512% of cells = 128K out of 25M
        vals = np.linalg.eigvals(A)
        e = np.max(np.abs(vals))
        self.A = (A/e) * self.radius            # paper: scaled by 0.1/max abs eigenvalue

    # Create matrix 'Win' = input weights
    # Originally part of train_reservoir()
    def _create_Win(self):
        D   = self.D
        C   = self.n_chan
        q   = int(D/C)                  # size of each input channel; paper: 5K/8=625
        Win = np.zeros((D, C))          # fixed input weights; paper: 5Kx8

        # initialize Win with partition structure: i loops input channels
        for i in range(C):
            np.random.seed(seed = i)
            Win[i*q:(i+1)*q, i] = self.sigma * (-1 
                                + 2 * np.random.rand(1, q)[0])
            # tall, sparse block matrix of fixed input weights; uniform random in [-1,1)
            # block i column i = sigma*(vector of q random numbers); rest are zeros
            # Win assigns each input to one reservoir partition consisting of q nodes
        self.Win = Win

# %%
# Run_PureESN -- run simple ESN models, no SINDy
#   Copy and modify for more complex models)
#   This is an apex class (no dependent classes); OK to copy and modify interface
class Run_PureESN:
    def __init__(self, name, data_model, ESN, Theta, Regress,
                 save=True, report=0):
        self.name    = name             # for documentation
        self.datam   = data_model       # provides training and testing data
        self.ESN     = ESN              # layer 1, derived from class ESN
        self.Theta   = Theta            # layer 2, derived from class Theta
        self.Regress = Regress          # layer 3, derived from class Regress
        self.n_train = 0                # updated in train()
        self.n_pred  = 0                # updated in predict()
        self.output  = None             # created in predict() -> model accuracy
        self.states  = None             # created in predict() -> model efficiency
        self.save    = save             # flag: save data thru-put arrays
        self.report  = report           # print messages up to this level of loops

    # Initialize the ESN-SINDy hybrid model
    def prepare(self, report=False):
        self.ESN.generate()
        # Save transient arrays (over-written if multiple models)
        self.Win = self.ESN.Win.copy()
        self.A   = self.ESN.A.copy()

    # Train model using input array 'data'; return final state vector r(n_train)
    def train(self, n_train, beta=None, report=False):
        '''Train model; data row=input channel, col=time; returns r(n_train)'''
        data = self.datam.data(0, n_train)         # grab training time steps

        # Simple model, batched: each layer processes entire data block in one call...
        r    = self.ESN.train(data, n_train)
        X    = self.Theta.train(r, report=report)
        Wout = self.Regress.train(X, data, beta=beta, report=report)
        # Iterative models will need to move Theta and Regress training into a loop
        self.n_train = n_train                  # set flag for .predict()
        if report>0:
            W00 = Wout[0,0]
            print(f'Run_PureESN.train(): Wout is {np.shape(Wout)}, [0,0]={W00:.4f}')
        # Save transient arrays (over-written if multiple models)
        if self.save:
            self.r    = r.copy()
            self.X    = X.copy()
        self.Wout = Wout.copy()
        return Wout
    
    # Predict a trajectory. Notation: in Chat 2020, X was r~ or x_aug.
    # r is single state vector, trajectory (array of many states), or None
    def predict(self, n_predict, r=None, report=0):
        '''Prediction trajectory; saves output and states for later analysis'''
        if report>0: print(f'Run_PureESN.predict({n_predict=}, {r=})')
        assert self.n_train > 0, 'Call train() before predict()'
        C  = self.ESN.n_chan                    # input channels
        D  = self.ESN.D                         # reservoir size
        tP = n_predict                          # prediction time steps
        
        if report>1: print(f'Run_PureESN.predict({tP=}); {C=} {D=} {tP=} {r=}')
        output = np.zeros((C, tP))              # for model performance plots
        states = np.zeros((D, tP))              # for internal analysis only
        if r is None:
            r = self.ESN.predict(None)          # retrieve final training state
            assert r is not None, 'ESN_Model.predict: ESN.predict()->None'
        for t in range(tP):
            X   = self.Theta.run(r, report)     # regression input vector
            out = self.Regress.predict(X)
            output[:,t] = out                   # predicted trajectory
            r   = self.ESN.predict(r, out)
            states[:,t] = r
            if report>1: print(f'{t=}: {out[0]:.4f} {out[1]:.4f} {out[2]:.4f}...')
        self.n_pred = tP
        # Save transient arrays (over-written if multiple models)
        self.output = output
        if self.save:
            self.states = states

    # Retrieve ground truth trajectory to complement .predict()
    def truth(self):
        assert self.n_train is not None and self.n_pred is not None, \
               'Run*: no n_pred: call predict() before truth()'
        return self.datamod.data(self.n_train, self.n_pred)

    # Display parameters
    def show(self):
        print(f'ESP_Model {self.name}: data model={self.datamod.name}.',
              f'Layers: esn={self.ESN.name}, theta={self.Theta.name},',
              f'regress={self.Regress.name}, n_train={self.n_train},',
              f' n_pred={self.n_pred}')

    # Assess performance
    def report(self, options):
        print('In progress. Add analytics')
    
# %%
# Performance metrics
# Confidence intervals for maintaining accuracy, based on multiple channels
# Single model
def ci1(array, ci_1side):
    '''One-sided confidence intervals; input = array of times to event'''
    # FUTURE: check math for time-to-ievent data; use poisson distribution?
    assert 0 <= ci_1side < 1
    v    = array.flatten()
    conf = 1 - 2*(1-ci_1side)           # convert 2-sided CI to 1-sided
    df   = len(v) - 1                   # degrees of freedom
    mean = np.mean(v)
    sem  = st.sem( v)                   # std error of the mean
    if sem > 0:
        ci = st.t.interval(conf, df=df, loc=mean, scale=sem)  # t-test
    else:
        ci = [mean, mean]               # all times are the same
    return ci[0]

# Stats for a single model and multiple channels
def stats1(truth, pred, conf=0.95):
    '''Compute pseudo R^2 and its decay; args are array(channel,time)'''
    (nc,nt) = np.shape(truth)       # data channels, time periods
    sst = np.zeros([nc, nt])        # truth dot prod, as f(t)
    sse = np.zeros([nc, nt])        # prediction error: sum of (t-p)^2
    R2  = np.zeros([nc, nt])        # R^2 as it evolves over time
    # levels of predictive accuracy (R^2 = coeff of det = 1-MSE/Var)
    levels = np.array([0.0, 0.5, 0.9, 0.99])
    # channel prediction accuracies (tr) and their confidence intervals
    tr_lev = np.zeros([len(levels),nc], dtype=int) # channel time to r<level
    ci_lev = np.zeros([len(levels)],    dtype=int) # lower bound conf interval
    # get truth sum of squares relative to mean
    for c in range(nc):
        truec = truth[c,:] - np.mean(truth[c,:])
        for t in range(nt):
            # true trajectory SS relative to mean
            sst[c,t] = sst[c,t-1] + truec[t]**2
    # get prediction error sum of squares
    for c in range(nc):
        for t in range(nt):
            # predicted trajectory SS relative to actual
            sse[c,t] = sse[c,t-1] + (pred[c,t]-truth[c,t])**2
            # r2 is pseudo R^2; may be negative
            #   r2 in [0,1] = proportion of truth variance explained by model;
            #   r2 < 0 = model diverges; prediction is worse than just using
            #       the average of truth, ie, model gives an anti-prediction.
            r2 = 1 - sse[c,t]/sst[c,t]
            R2[c,t] = r2
            for i, lev in enumerate(levels):
                if r2 > lev: tr_lev[i,c] = t+1  # time to R^2 < cutpoint
    # summary: confidence interval lower bound for each error threshold level
    for i, lev in enumerate(levels):
        ci_lev[i] = np.floor(ci1(tr_lev[i,:], conf)).astype(int)
    # eg, 95% sure we maintain 99% accuracy to this time step
    return tr_lev, ci_lev, R2   # levels x channels, levels, channels x time
    # end function stats1()
