# -*- coding: utf-8 -*-
"""
Created on Tue May 20 20:26:45 2025
@author: buddy
Contents: New versions of core classes.
Import after main library; replaces Theta and Regress.

25 May 2025 21:30 (v2.2) - Bug fixes & output improvements for HS support:
    Theta.tune() - Was learning in the wrong direction  *** UNSTABLE, AVOID ***
    Theta.add() - Add relevance from Regress.relevance() to improve accuracy of
        identification of low-performing rows to replace.
    Regress.relevance() - Fixed logic and math errors; relevance now defined as
        average magnitude of normalized regression coefficients, reflecting
        the true contribution of each variable to prediction accuracy.
    Regress.cross_rel() replaces cross_cov() - Now uses estimated regression
        coefficients, same logic as Regress.relevance() and way more useful.
    Regress.train() now normalizes input data around the mean; this improves
        regression performance when inputs are highly assymetric, as we have
        with many polynomial terms.
"""
import numpy as np
#import scipy.stats  as st
import scipy.sparse as sparse
import matplotlib.pyplot as plt

# %%

# Helper function to tidy up arrays produced by matrix operations
def vector(m):
    '''Convert 1xN or Nx1 matrix to vector(N,). Result.ndim: np.sum(m.shape>1)'''
    assert isinstance(m, np.ndarray)
    return np.asarray(m).squeeze()

# %%
# Regress (layer 3) -- regress data on theta(r) to train, compute next r to predict
class Regress:
    def __init__(self, name, dmodel, beta=0.0001):
        self.name    = name             # for documentation
        self.dname   = dmodel.name      # for documentation
        self.n_chan  = dmodel.dim(0)    # data partitions (if used; else ignored)
        self.beta    = beta             # weights L2 norm to manage colinearity
        self.Wout    = None             # created by train()
        self.report  = 0                # print messages up to this level of loops
        self.dLdX    = None             # flag: call .grad() to find gradients

    # Terminology: our X = p from Theta.
    # Find Wout; save loss terms
    def train(self, X, data, beta=None, report=0, norm=True):
        '''Return Wout. X is input data (X=p=Theta(r), data is "truth"'''
        if beta is None: beta = self.beta
        self.norm = norm                        # False to replicate version 1
        C = data.shape[0]                       # input channels
        E = X.shape[0]                          # input data
        T = 1 if X.ndim==1 else X.shape[1]      # time steps
        
# =============================================================================
        ## Note that X lacks constant term (vector of ones), so if inputs are
        ##   not normalized to mean 0, then regression must find complementary
        ##   terms to balance each other to prevent a constant prediction bias.
        ## Squared terms in T_1 are always >0, so their coefficients must have
        ##   balancing signs & magnitudes: adds burden to model performance.
# =============================================================================
        # Normalize input data (recommended)
        if norm:
            Xmean = np.mean(X, axis=1)
            X = (X - Xmean.reshape(-1,1))

        idenmat = beta * sparse.identity(np.shape(X)[0])

        # Covariance matrix and regression coefficients
        cov = np.dot(X, X.transpose())          # (ExE)
        U = cov + idenmat                       # (ExE)
        Uinv = np.linalg.inv(U)                 # (ExE)
        Wout = np.dot(Uinv, np.dot(X, data.T))  # (ExC)
        Wout = Wout.T                           # (CxE)
        Xstd = np.sqrt(np.diag(cov))            # (E)  Only valid if norm=True

        # Save Wout, sizes, pointers to passed data to sync with grad()
        self.Xstd = Xstd
        self.Wout = Wout
        self.C = C
        self.E = E
        self.T = T
        self.X = X
        self.data = data
        self.dLdX = None  # indicator that gradients etc are now obsolete
        return self.Wout

    # Gradient of loss w.r.t. input variables. Call train() before grad().
    # ======== Outputs... ================================================
    # mse_t  = Total MSE over time and channels (average of mse_c)
    # mse_c  = MSE for each channel (main diagonal of mse_cc)
    # mse_cc = Cross-channel MSE. High mse_cc[i,j] = likely i-j dynamics.
    # dLdX   = Backprop loss for Theta layer weight tuning
    # dLdX_sd= High value implies term's impact fluctuates, may be driven
    #          by multiple dynamics features or interactions: add terms.
    # ====================================================================
    def grad(self, gigs=8):
        '''Compute loss and gradient stats. Input data, truth go in X and data;
        gigs controls computation of gradient variability (heat) in dLdX_sd:
        gigs=0 to suppress computation, else gigs=memory cap before switching
        to looping inputs (slower but saves memory).'''
        C = self.C                      # num input channels
        E = self.E                      # num input data (output from Theta)
        T = self.T                      # num time steps
        X = self.X                      # input data  (ExT)
        data = self.data                # truth data  (CxT)
        Wout = self.Wout                # reg weights (CxE)

        # Loss gradient starts with prediction errors by channels and time
        y_pred = np.dot(Wout, X)                # predicted outputs     (CxT)
#???===========================================================================
#??? Should be flipped: dev = data - y_pred
#??? After class paper, see if flipping the sign helps ???
#??? NOPE, we're good, this is multiplied t=by dy_pred/dvar, which also flips sign
        dev    = y_pred - data                  # loss=prediction error (CxT)
#???===========================================================================
        
        # Error (L2 norm^2, 2*Loss): total and by input channel
        mse_cc= np.dot(dev, dev.T) / T          # Cross-Channel MSE     (CxC)
        mse_c = np.diag(mse_cc)                 # MSE by Channel        (C)
        mse_t = np.sum( mse_c) / C              # MSE Total, scalar     (.)
        
        # Gradients relative to weights and data
        dLdWout = np.dot(dev, X.T) / T          # mean(error*X)         (CxE)
        devW = np.dot(Wout.T, np.sum(dev,1))    #                       (Ex1)
        dLdX = vector(devW) / (T*C)             # mean(error*Wout)      (E)
        
        # Variability of dLdX (heat); fast if enough memory, else loop (slower)
        if gigs > 0:
            if E*T*8 < gigs*1e9:
                stdev = np.std(np.dot(Wout.T, dev), 1)  # sd on rows of (ExT)
            else:
                stdev = np.zeros(E)
                for i in range(E):
                    stdev[i] = np.std(np.dot(Wout.T[i,:], dev[:,i]))
            dLdX_sd = stdev
        dLdX_sd = vector(dLdX_sd)
            
        # Save and return results
        self.mse_cc = mse_cc                    # cross-channel MSE     (CxC)
        self.mse_c  = mse_c                     # MSE by channel        (C)
        self.mse_t  = mse_t                     # MSE total             (.)
        self.dLdWout= dLdWout                   # mean(error*X)         (CxE)
        self.dLdX   = dLdX                      # mean(error*W)         (E)
        self.dLdX_sd = dLdX_sd                  # std dev(error*W)      (E)
        return mse_t, mse_c, mse_cc, dLdX, dLdX_sd

    def relevance(self):
        '''Return |normalized Wout|; impact on prediction in units of st.dev'''
        Wout = self.Wout                        #                       (C,E)
        Xstd = self.Xstd                        #                       (E,1)
        Wabs = np.mean(np.abs(Wout), axis=0)    # average magnitude     (1,E)
        term_rel = vector(Wabs) * vector(Xstd)  # normalize             (E,)
        return term_rel

    def cross_rel(self, step=1):
        '''Can new variable pairings reduce MSE? Analyse residuals for hints.
        Returns 2 arrays, sorted by relevance: (1) indices in theta outputs (p),
        (2) relevance of their product as |w|, w=impact on residuals.
        Optional step arg samples (100/step)% of data for faster execution.'''
        X    = self.X           # input data (normalized to mean=0 in train)
        data = self.data        # predicted data (truth)
        Wout = self.Wout        # regression weights                    (C,E)
        C    = Wout.shape[0]    # channels
        E,T  = X.shape          # number of variables and times
        P    = int(E*(E+1)/2)

        ### print(f'--- cross_rel: {C=}, {E=}, {P=}, {T=}') #???
        # Triangularize data to save memory and simplify matrix ops
        p     = np.zeros((P,1), dtype=float)
        index = np.zeros((2,P), dtype=int  )
        inext = 0
        istop = E
        for i in range(E):
            index[0, inext:istop] = i
            index[1, inext:istop] = np.arange(i,E)
            inext = istop
            istop += (E-i-1)
        
        # 'p'=input vars, 'y'=predicted=deviation (error)
        # Accumulators for sums; ecov holds sum of p * squared error
        p_1 = np.zeros((P,1))                   # just p, for mean      (P,1)
        p_2 = np.zeros((P,1))                   # p^2                   (P,1)
        p_y = np.zeros((P,1))                   # p*dev                 (P,1)
        y_1 = 0.0                               # errors, for mean      (.)
        y_2 = 0.0                               # SSE                   (.)

        for t in np.arange(0,T,step):
            # Reshape data vectors into matrices so matrix ops work properly
            Xt = X[   :,t].reshape(-1,1)        # input vector          (E,1)
            Dt = data[:,t].reshape(-1,1)        # truth vector          (C,1)

            # Triangularize data for current time step
            inext = 0
            istop = E
            for i in range(E):
                p[inext:istop,0] = Xt[i:E,0] * Xt[i,0]  # new terms     (P,1)
                inext = istop
                istop += (E-i-1)

            dev = np.dot(Wout, Xt) - Dt         # prediction error      (C,1)
            devs= np.sum(dev)                   # sum over channels     (.)
            
            # Accumulate sums
            p_1 += p                            # used for std dev      (P,1)
            p_2 += p * p                        # element-wise mult     (P,1)
            p_y += p * devs                     # all vars x channels   (P,1)
            y_1 += np.sum(dev)                  # deviation, for mean   ()
            y_2 += np.dot(dev.T, dev)           # sum SSE over channels (1,1)

        # Single-variable normed regression coefficients, by var and channel        
        N = (T // step)
        numer = vector(p_y - p_1*y_1/N)/N/C     # numerator, all ch's   (P,1)
        denom = vector(p_2 - p_1*p_1/N)/N       # denominator = var(p)  (P,1)
        stdev = np.sqrt(denom)                  # std dev = root(var)   (P,1)

        # Relevance = reg coefficient * stdev = numer/stdev
        try:
            wabs = np.abs(numer) / stdev        # rel = |reg|*stdev     (P,1)
        except:
            print('cross_rel(): invalid divide; denom=\n{denom[:20]...')
        srt = np.argsort(-wabs).ravel()                         #       (P,)
        idx = index[:,srt].copy()                               #       (2,P)
        rel = wabs[srt].copy().ravel()                          #       (P,)
        return (idx, rel)

    # Compute next step of predicted system trajectory
    def predict(self, X, report=0):
        out = np.squeeze(np.asarray(np.dot(self.Wout, X)))
        return out

    # Display parameters
    def show(self):
        print(f'Regression layer {self.name}: data model={self.dname}, beta={self.beta}')

# %%
# Theta -- Generalized Theta layer: matrix W = list of polynomial exponents
#          Call a set_W_* method to find E and initialize W
# =============================================================================
# #??? WARNING: reg IS NOW REQUIRED - MUST CALL Regress < Theta
#??? Dropped 'name' - not found in any code --> never used
# =============================================================================
class Theta():
    def __init__(self, esn, reg, lr=0.01, report=0):
        self.esn    = esn               # in case we need data info
        # Array dimensions...
        self.C = esn.n_chan             # data partitions (if used; else ignored)
        self.D = esn.D                  # size of reservoir
        self.S = self.D // self.C       # size of each input channel
        self.reg = reg                  # Regress layer for add()
        #???
        self.lr     = lr                # initial learning rate for tune()
        self.report = report            # log up to this level of loops

    def set_W_Ti(self, T_i):
        '''Initialize W for classic ESN (square matrix)'''
        C = self.C
        D = self.D
        self.E = D
        S = self.S
        W = np.zeros((D,D), dtype=complex)
        if T_i == 0:
            for i in range(D): W[i,i] = 1
            self.W = W
            return
        for i in range(D):
            if np.mod(i,2) == 1:        # odd-indexed nodes: no transformation
                W[i,i] = 1
            else:                       # even nodes: select a basis function
                val=1
                match T_i:
                    case 1: j=i;   k=i;  val=2  # square self
                    case 2: j=i-2; k=i-1        # 2 preceding nodes
                    case 3: j=i-1; k=i+1        # 2 neighboring nodes
                    case 4: j=i-C; k=i+C        # neighboring channels
                    case 5: j,k=np.random.randint(S*(i//S  ),
                                                  S*(i//S+1), 2)
                    case 6: j,k=np.random.randint(0,D,2)
                    case 7: j=i-2*C; k=i-C      # 2 preceding channels
                    case _:
                        print(f'Set_W_i_{T_i}: unknown index {T_i=}')
                        break
                # if over shot: go around to other end of the ring
                if j <0: j+=D
                if k <0: k+=D
                if j>=D: j-=D
                if k>=D: k-=D
                W[i,j] = val
                W[i,k] = val
            # end else even node
        self.W = W

    def set_W_SINDy(self, degree=2):
        '''Create blank poly term matrix. n_out=output size; square if None'''
        D = self.D
        E = D if degree==1 else int(D*(D+3)/2)   # poly terms to degree 1 or 2
        if degree > 2:
            print(f"Theta.init_W_HS({degree=}): too high, using degree 2")
        # Create W...
        W = np.zeros((E,D), dtype=complex)
        for i in range(D):
            W[i,i] = 1
        if degree >= 2:
            for i in range(D):
                W[D+i,i] = 2
            k = 2*D
            for i in range(D-1):
                for j in range(i+1,D):
                    W[k,i] = 1
                    W[k,j] = 1
                    k += 1
            assert k == E
        # Save new internal variables
        self.E = E
        self.W = W

    def set_W_HS(self, diag=None, n_out=None):
        '''Create blank poly term matrix. n_out=output size; square if None'''
        D = self.D
        self.E = D if n_out is None else n_out
        W = np.zeros((self.E, D), dtype=complex)
        if diag is not None:
            for i in range(min(D, self.E)):
                W[i,i] = 1+0j
        self.W = W

#??? =============================================================================
#??? Hypothesis: rounding error compromises accuracy on ESN models
#??? =============================================================================
    # Terminology: Output is p, for polynomial expansion of input r
    def run(self, r, p_type='real', report=0):
        '''Return output p = exp(W log(r)).real; use p_type to keep complex'''
        # r = current ESN state(s) from ESN.output; D rows, T column(s)
        # returns either vector or matrix, same columns as r
        with np.errstate(divide='ignore'):      # suppress /0 warning
            logr = np.log(r.astype(complex))    # -inf if r==0; j!=0 if r<0

        logp = np.dot(self.W, logr)             # log of output p
        p    = np.exp(logp)
        p    = np.where(np.isnan(p), 0, p)      # turn input 0's back to 0
        
# =============================================================================
#         # Assess magnitude of imaginary terms, save for diagnostics
#         self.imag_mean = np.mean(abs(p.imag))
#         self.imag_max  = np.max(abs(p.imag))
# =============================================================================
        if p_type=='real':
            p = p.real

        # Save in and out vectors for analysis; grad() uses r and logp
        # r may be column vector, but saved internally as 2d array
        self.r  = r                             # pointer to data, not copy
        self.logp = logp
        self.p  = p
        self.T  = 1 if r.ndim==1 else r.shape[1]     # number of time steps
        return p

    # -------------------------------------------------------------------------
    #                 HS and NN Tools - gradW, gradR, tune_W
    # -------------------------------------------------------------------------
    
    # Gradients of loss w.r.t. W (sum over time steps) and r (by time)
    # Terminology: lp=W log(r), p=exp(lp) = Theta output = Regress input X
    # Uses data from most recent calls to Theta.run() and Regress.train()
    def gradW(self, dLdP):
        '''Return loss gradient w.r.t. W. Used to learn weights in W.
        Input arg is dL/dP from the next layer (Regress).'''
        # Example: dLdW = theta.gradW(regress.grad())
        dPdW = np.dot(self.p, self.r.T)   # (ExT)(TxD)-->(ExD), matches W
        ### print(f'**** gradW: {self.p.shape=}, {self.r.shape=}, {dPdW.shape=}, {dLdP.shape=}') #???
        return dLdP*dPdW                  # (ExD)

    def gradR(self, dLdP):
        '''Return loss gradient w.r.t. r. Used by previous layer (ESN) to learn
        reservoir parameters. Input is dL/dP from next layer (Regress)'''
        # Example: dLdR = theta.gradR(regress.grad())
        dPdR = np.dot(self.W.T, self.p) / self.r    # elementt-wise division
        return dLdP*dPdR
    
    # Main function to adjust exponents (W)
    # User must call Regress.train() and grad(), and pass Regress.dLdP
    #######################################################################???
    # 26May2025 - Added L2 norm regularization to damp the divergence of
    #   both Theta.W and Regress.Wout unless Regress.beta is extremely large
    #   (beta=0.1 with small test model, D=40 and iterations=10).
    #######################################################################???
    def tune_W(self, dLdP, sparse=True, l2weight=0.001, lrlr=0.001):
        '''Adjust exponents in W using loss gradient backpropagation.
        dLdP is Regress.dLdX; call reg.train() and reg.grad() to compute it.
        sparse=True to adjust non-zero weights, sparse=False to adjust all.
        Initial learning rate lr will self-adjust over time.
        lrlr is learning rate for the learning rate=time scale for d(lr)/dt.'''

        # Use dL/dP and l2 norm to compute dL/dW
        wr = self.W.real
#???    wi = self.W.imag
        dRegLoss_dW = self.gradW(dLdP)          # gradients for all W, even 0's
#??? ##################### CONFIRM - WHY THE SQRT? NOT USED IN REG LOSS ##############
#??? AND DIVIDING SEEMS TO NEGATE ITS EFFECT - LIKELY BUG 
#??? ASK MATT TO REVIEW
        dL2Norm_dW  = wr/np.sqrt(np.sum(wr**2)) #???+ \
        #???              wi/np.sqrt(np.sum(wi**2))j
        dLdW = dRegLoss_dW + l2weight * dL2Norm_dW
#??? ##################### CONFIRM #################################

        # Get lagged lr and gradient (initialize to 0 if first call)
        lr = self.lr
        try:    lag = self.dLdW
        except: self.dLdW = lag = np.zeros_like(dLdW)
        
        # Adjust learning rate; up if current and lag gradients in same
        #   direction, down if in opposite directions: dlr/dt ~ avg product
        grad_prod = np.dot(dLdW.ravel(), lag.ravel().T)
        scale     = np.dot(dLdW.ravel(), dLdW.ravel().T)
        
        # Adapt learning rate, as simple ODE with internal equilibrium at .5
        #   and grad_prod as forcing term: grad_prod<0 = oscillate = too fast
        with np.errstate(invalid='ignore'):
            dlr_dt = 0.5 - lr + grad_prod/scale
        if np.isnan(dlr_dt):
            dlr_dt = 0.0
        lr = lr + lrlr * dlr_dt
        
        # Adjust weights
        W = self.W
        if sparse: self.W = np.where(W==0, W, W - lr*dLdW)
        else:      self.W = W - lr*dLdW
        # Save lags for next iteration
        self.lr = lr
        self.dLdW = dLdW

    # -------------------------------------------------------------------------
    #          HS Term Operations - add, delete, duplicate, replace
    # -------------------------------------------------------------------------

    def add(self, ind, exp=1):
        '''Add polynomial term to W, from list of indexes and optional exponents.
        Default exp is 1; if scalar, the same exponent is used for all indexes.
        If provided, relevance or self.reg will improve row recylcling.
        Replaces first empty row; if none, row with smallest absolute sum.
        Returns index of added term.'''
        n = len(ind)
        if np.isscalar(exp): exp = [exp]*n
        
        # Find a blank row (FUTURE FEATURE: or expand W with new row)
        relevance = self.reg.relevance()
        if relevance is None:
            relevance = np.sum(np.abs(self.W), axis=1)
        
        # Recycle the selected under-performing row
        first_row = np.argsort(relevance, kind='stable')[0]
        self.replace(first_row, ind, exp)
        return first_row
    
    def delete(self, row):
        '''Remove row from active terms, replacing W[row,:] with zeros'''
        ## flag old row as inactive in the term reservoir
        self.W[row,:] = [0]*self.D
        
    def duplicate(self, row):
        '''Duplicate the existing term at row. Returns index to new row.'''
        # Identify best element to bifurcate
        W = self.W.real
        elements = np.nonzero(W[row,:])[0]
        if len(elements)==0:
            W[row, row] = 0.5
            elements = np.nonzero(W[row,:])[0]
        midways  = np.zeros(len(elements))
        for i,idx in enumerate(elements):
            w = W[row,idx] % 1
            midways[i] = w*(1-w)
        maxdev = np.argsort(midways, kind='stable')   # deviations from int
        #???  ================================
        if len(elements)==0:
            print(f'duplicate({row=}) EMPTY ROW')
        bif_id = elements[maxdev[-1]]                 # column index of target
        # Copy row, moving bifurcation element down on old and up on new
        w1 = np.floor(W[row, bif_id])                 # old term rounds down
        w2 = np.ceil( W[row, bif_id])                 # new term rounds up
        self.W[row,bif_id] = w1
        values = W[row,:].copy()
        values[bif_id] = w2
        # Add the copy
        new_row = self.add(elements, values[elements])
        # If rows identical (all W's integer), bump main diagonal to break tie
        if w1==w2:
            self.W[new_row, new_row] += 1
        return new_row

    def replace(self, row, ind, exp):
        '''Replace row with new term, defined by lists of indexes and exponents'''
        assert 0 <= row < self.E, f'{row=} out of range in Theta.replace()'
        Wrow = np.zeros(self.D)
        for i,idx in enumerate(ind):
            assert 0 <= idx < self.D, f'index {i}={idx} out of range in Theta.replace()'
            Wrow[idx] = exp[i]
        ## flag old row as inactive in the term reservoir
        ## add new row to reservoir, or flag as active if already exists
        self.W[row,:] = Wrow

    # -------------------------------------------------------------------------
    #          Support functions
    # -------------------------------------------------------------------------

    def find_splits(self, rel=True):
        '''Sort rows on duplication potential; return sorted index, row scores.
        To include regression relevance, set rel=True'''
        w = self.W.real.copy() % 1      # w[i,j] in [0,1)
        w = w*(1-w)*4                   # scaled to [0,1]
        score = np.sum(w, axis=1)
        if rel: score *= self.reg.relevance()
        index = np.argsort(-score)
        return index, score

    def dim(self, index=None):
        '''return dimensions of stored data array'''
        try:    return np.shape(self.W)
        except: return [0,0]

    def show(self, scale=200):
        if not hasattr(self,'W'): print('Theta not yet initialized'); return
        W = self.W
        C = self.C
        S = self.S
        E,D = np.shape(W)
        print(f'Theta: W is size {E=} x {D=}; input {C=} x {S=}; see plot')
        fs = 8/max(D, E)        # figure scale: inches per cell
        xylist = []
        clr=[]
        for i in range(E):
            c = ((i%5)/5, (i%20)/20, (i%100)/100)     # colors vary by row
            for j in range(D):
                if W[i,j] != 0:
                    xylist.append([i,j,W[i,j]])
                    clr.append(c)
        xy = np.array(xylist).astype(int)
        
        bigdots = (max(xy[:,2]) + np.mean(xy[:,2]))/2
        ds = scale*fs/bigdots # dot scale: max cell fill available space
        plt.figure(figsize=(fs*D, fs*E))
        plt.scatter(xy[:,1], xy[:,0], ds*abs(xy[:,2]), c=clr)
        plt.ylim(E-0.6, -0.4)
        plt.xlim(-0.4, D-0.6)
        plt.show()
