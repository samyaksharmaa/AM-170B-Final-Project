# Heuristic ESN hybrid - plots

import numpy as np
import pandas as pd                     # added 3jun2025 for HS .csv reading
import matplotlib.pyplot as plt
from   scipy.stats import gaussian_kde

#%%
# Utility functions - extract fixed and line-specific fields from HS_ESN.run*()

def unpack_fixed(rtruth):
    '''Extract result fields that remain fixed for a set of predictions'''
    truth, name, D, E, T = [rtruth[k] for k in ['traj','name','D','E','T']]
    return (truth, name, D, E, T)

def unpack_lines(result):
    '''Extract result fields unique to each prediction in a set; call in loop'''
    traj, model, name = [result[k] for k in ['traj', 'model', 'name' ]]
    return (traj, model, name)

#%%
class HS_ESN_plots:
    def __init__(self, hs):
        self.hs = hs
        self._plot_id = 0

    def report(self, predictions, truth, channel=0, save=False):
        self.plot_theta_forecasts(predictions, truth, channel, save=save)
        #self.plot_accuracy(pa, ci, r2, save=save)  #??? moved to recycle_bin.py
        self.plot_error_vs_time(predictions, truth, channel, save=save)
        self.plot_prediction_pdfs(predictions, truth, save=save)
        self.plot_combined_l2_trajectory(predictions, truth, save=save)

    @staticmethod
    def T_labels():
        # index assignments: ESN uses 0-7, HS uses 10-12 (add 10 to HS ID)
        labels = ['T_0: No transformation', 'T_1: Square even states',
                  'T_2: 2 previous neighbors', 'T_3: 2 nearest neighbors',
                  'T_4: Neighboring channels', 'T_5: Random in-chan nodes',
                  'T_6: Random from anywhere',
                  'T_7: Combo: T_3 T_4 T_6', '', '',
                  'HS from diagonal=0', 'HS from diagonal=1',
                  'HS from diagonal=2']
        colors = ['#dd0000', 'c', 'g', 'b',         # ESN: I, classic, Ashesh
                  '#a000ff', '#70c000', '#f08000',  # ESN: our variations
                  '#ff00f0', 'k', 'k',              # unused
                  'k', 'k', 'k']                    # HS, diagonal = 0, 1, 2
        #??? Add HSN colors to the above if you want something other than black
        abbrev = ['T_0', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'na', 'na',
                  'HS0', 'HS1', 'HS2']
        # misc
        accuracy = ['0%', '50%', '90%', '99%']
        return (labels, colors, abbrev, accuracy)

#%%
    def plot_theta_forecasts(self, results, rtruth, channel=0,
                             title='Forecasts with ', line_labels='long',
                             error_threshold=0.3,  save=False):
        # Extract fixed result fields (for the whole set of predictions)
        truth, name, _,_,_ = unpack_fixed(rtruth)
        # Line attributes, indexed by model
        labels, colors, abbreviations, _ = self.T_labels()

        plt.figure(figsize=(12, 6))
        n_predict = truth.shape[1]
        time_steps = np.arange(n_predict) / 200
        plt.plot(time_steps, truth[channel, :], 'k--', lw=2, label='Truth')
        for result in results:
            # Extract result fields for each line
            pred, model, line_name = unpack_lines(result)
            color = colors[model]
            match line_labels:
                case 'short':  label = abbreviation[model]
                case 'custom': label = line_name
                case _:        label = labels[model]

            e_t = np.linalg.norm(truth - pred, axis=0) \
                / np.mean(np.linalg.norm(truth, axis=0))
            exceed_idx = np.argmax(e_t > error_threshold) \
                if np.any(e_t > error_threshold) else None
            plt.plot(time_steps, pred[channel, :], color=color, label=label,
                     alpha=0.8)
            if exceed_idx:
                plt.axvline(x=time_steps[exceed_idx], color=color, 
                            ls=':', lw=2.5, alpha=0.8)
        plt.title(f'{title}{name} | L₂ Threshold ε={error_threshold}')
        plt.xlabel('Time (MTU)')
        plt.ylabel(f'X_{channel}(t)')
        plt.legend(loc='upper right')
        plt.ylim(-4,4)
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.hs.name}_fig_{self._plot_id}.png')
            self._plot_id += 1
        plt.show()
    # end plot_theta_forecasts()

### def plot_accuracy(self, pa, ci, r2, midx=None, ...)
### removed 01jun2025, copied to recycle_bin.py (edwin)

#%%
    def plot_error_vs_time(self, results, rtruth, channel=0, 
                           title='Forecasts with ', line_labels='long',
                           error_threshold=0.3, save=False):
        # Extract fixed result fields (for the whole set of predictions)
        truth, name, _,_,_ = unpack_fixed(rtruth)
        # Line attributes, indexed by model
        labels, colors, abbrevs, _ = self.T_labels()

        n_time_steps = truth.shape[1]
        time_steps = np.arange(n_time_steps) / 200
        plt.figure(figsize=(12, 6))

        # Error denominator
        mean_truth = np.mean(np.linalg.norm(truth, axis=0))
        for result in results:
            pred, model, line_name = unpack_lines(result)
            # Error for each model
            err = (np.linalg.norm(truth - pred, axis=0))/mean_truth
            plt.plot(time_steps, err, color=colors[model],
                     label=f'{abbrevs[model]} error')

        plt.axhline(y=error_threshold, color='k', linestyle='--', linewidth=1,
                    label=f'Threshold (e={error_threshold})')
        plt.title('Relative L2 Error vs. Time')
        plt.xlabel('Time (MTU)')
        plt.ylabel('Relative $L_2$ Error $e(t)$')
        plt.legend()
        plt.ylim(0,2)
        plt.tight_layout()
        if save:
            plt.savefig(f'fig_{self._plot_id}.png')
            self._plot_id += 1
        plt.show()
    # end plot_error_vs_time()

#%%
    def plot_prediction_pdfs(self, results, rtruth, n_quartiles=4, save=False,
                             line_labels='long'):
        # Extract fixed result fields (for the whole set of predictions)
        truth, name, D, E, n_train = unpack_fixed(rtruth)

        # Plot attributes for each line
        labels, colors, abbreviations, _ = self.T_labels()

        plt.figure(figsize=(12, 8))
        truth_std = (truth - np.mean(truth)) / np.std(truth)
        kde_truth = gaussian_kde(truth_std.flatten())
        x = np.linspace(-4, 4, 1000)
        plt.plot(x, kde_truth(x), 'k--', lw=2, label='Truth')
        for result in results:
            pred, model, line_name = unpack_lines(result)
            color = colors[model]
            match line_labels:
                case 'short':  label = abbreviation[model]
                case 'custom': label = line_name
                case _:        label = labels[model]

            preds_std = (pred - np.mean(pred)) / np.std(pred)
            kde = gaussian_kde(preds_std.flatten())
            plt.plot(x, kde(x), color=color, label=label, alpha=0.7)

        plt.title('PDF Comparison: T₀-T₆ Predictions')
        plt.xlabel('Values')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.yscale('log')
        plt.ylim(1e-4, 1)
        plt.tight_layout()
        if save:
            plt.savefig(f'fig_{self._plot_id}.png')
            self._plot_id += 1
        plt.show()
    # end plot_prediction_pdfs()

#%%
    def plot_combined_l2_trajectory(self, results, rtruth, channel=0,
                                    error_threshold=0.3, save=False,
                                    line_labels='long'):
        '''Set line_labels='short' or 'custom' for abbreviations or hs.name'''

        # Extract fixed result fields (for the whole set of predictions)
        truth, name, D, E, n_train = unpack_fixed(rtruth)

        # Plot attributes for each line
        labels, colors, abbreviations, _ = self.T_labels()

        plt.figure(figsize=(12, 6))
        time_steps = np.arange(truth.shape[1]) / 200
        truth_l2 = np.linalg.norm(truth, axis=0)
        plt.plot(time_steps, truth_l2, 'k--', label='Truth (L₂ norm)', linewidth=2)

        for result in results:
            pred, model, line_name = unpack_lines(result)
            color = colors[model]
            match line_labels:
                case 'short':  label = abbreviation[model]
                case 'custom': label = line_name
                case _:        label = labels[model]
            pred_l2 = np.linalg.norm(pred, axis=0)
            plt.plot(time_steps, pred_l2, color=color, label=label, alpha=0.8)
            e_t = np.linalg.norm(truth - pred, axis=0) / np.mean(truth_l2)
            if np.any(e_t > error_threshold):
                exceed_idx = np.argmax(e_t > error_threshold)
                plt.axvline(x=time_steps[exceed_idx], color=color,
                            ls=':', lw=1.5)
        plt.title('Combined Trajectory (L₂ Norm Across All Channels)')
        plt.xlabel('Time (MTU)')
        plt.ylabel('$||X(t)||_2$')
        plt.legend()
        plt.ylim(0,10)
        plt.tight_layout()
        if save:
            plt.savefig(f'fig_{self._plot_id}.png')
            self._plot_id += 1
        plt.show()
    # end plot_combined_l2_trajectory()

#%%
    def plot_selected_trajectories(self, results, rtruth, channel=0,
                                   error_threshold=0.3, width=12, height=6,
                                   title='Predictions for ',
                                   line_labels='long'):
        '''Plot several trajectories vs truth. T_labels supplies line colors.
        results and rtruth are dictionairies from the HS_ESN.run_*() functions.
        Set line_labels='short' or 'custom' for abbreviations or hs.name'''

        # Extract fixed result fields (for the whole set of predictions)
        truth, name, D, E, n_train = unpack_fixed(rtruth)

        # Plot attributes for each line
        labels, colors, abbreviations, _ = self.T_labels()

        # Set up the plot, draw the 'truth' baseline
        plt.figure(figsize=(width, height))
        time_steps = np.arange(truth.shape[1]) / 200
        print(f'{time_steps.shape=}, {truth.shape=}, {channel=}') #???
        plt.plot(time_steps, truth[channel,:], 'k--', linewidth=2, label='Truth')
        
        # Loop and plot trajectories
        for result in results:
            # Extract result fields for each line
            pred, model, line_name = unpack_lines(result)
            color = colors[model]
            match line_labels:
                case 'short':  label = abbreviation[model]
                case 'custom': label = line_name
                case _:        label = labels[model]
            
            # Calculate error
            e_t = np.linalg.norm(truth - pred, axis=0) \
                / np.mean(np.linalg.norm(truth, axis=0))
            if np.any(e_t > error_threshold):
                exceed_idx = np.argmax(e_t > error_threshold)
            else:
                exceed_idx = None
            
            # Plot
            plt.plot(time_steps, pred[channel,:],
                     color=color, label=label, alpha=0.8)
            
            # Mark threshold crossing
            if exceed_idx:
                plt.axvline(x=time_steps[exceed_idx], color=color,
                           linestyle=':', linewidth=2)

        plt.title(f'{title}{name} | {D=}, {n_train=}')
        plt.xlabel('Time (MTU)')
        plt.ylabel(f'X_{channel}(t)')
        plt.legend()
        plt.ylim(-4,4)
        plt.tight_layout()
        plt.show()

#%%
    #??? Get argument logic from Brayan's original plotting code
    def plot_avg_l2_error(self, avg_errors, std_errors, channel=0,
                          error_threshold=0.3, width=12, height=6,
                          line_labels='long'):
        """Plot average L2 error across 30 ICs with CI"""

# =============================================================================
#         # Extract fixed result fields (for the whole set of predictions)
#         truth, name, D, E, n_train = unpack_fixed(rtruth)
# 
#         # Plot attributes for each line
#         labels, colors, abbreviations, _ = self.T_labels()
# =============================================================================

        plt.figure(figsize=(width, height))
        time_steps = np.arange(len(avg_errors[0])) / 200

        # Plot each method
        for result in results:
            # Extract result fields for each line
# =============================================================================
#             pred, model, line_name = unpack_lines(result)
#             color = colors[model]
#             match line_labels:
#                 case 'short':  label = abbreviation[model]
#                 case 'custom': label = line_name
#                 case _:        label = labels[model]
# =============================================================================

            mean = avg_errors[model]
            std  = std_errors[model]
            plt.plot(time_steps, mean, color=color, label=label)
            plt.fill_between(time_steps, mean-std, mean+std,
                             color=color, alpha=0.2)

            # Mark threshold crossing
            if np.any(mean > error_threshold):
                exceed_idx = np.argmax(mean > error_threshold) 
                plt.scatter(time_steps[exceed_idx], error_threshold,
                            color=color, s=100, marker='x')

        plt.axhline(error_threshold, color='k', linestyle='--', label=f'Threshold (ε={error_threshold})')
        plt.title(f'Average Relative L2 Error (30 ICs)\nD={D}, n_train={n_train}')
        plt.xlabel('Time (MTU)')
        plt.ylabel('Relative $L_2$ Error')
        plt.legend()
        plt.tight_layout()
        plt.show()
    # end plot_avg_l2_error()

#%%
    def plot_long_pdfs(self, results, rtruth, channel=0,
                       n_samples = 1000, width=12, height=6,
                       line_labels='long'):
        """Plot PDFs for long predictions (100k steps)"""

        # Extract fixed result fields (for the whole set of predictions)
        truth, name, D, E, n_train = unpack_fixed(rtruth)

        # Plot attributes for each line
        labels, colors, abbreviations, _ = self.T_labels()
        plt.figure(figsize=(width, height))
        
        # Downsample
        stride = len(truth[0]) // n_samples
        truth_down = truth[:, ::stride]

        # Standardize
        truth_std = (truth_down - np.mean(truth_down)) / np.std(truth_down)
        
        # Plot truth
        x = np.linspace(-4, 4, 1000)
        kde_truth = gaussian_kde(truth_std.flatten())
        plt.plot(x, kde_truth(x), 'k--', label='Truth')

        # Plot predictions
        for result in results:
            pred, model, line_name = unpack_lines(result)
            color = colors[model]
            match line_labels:
                case 'short':  label = abbreviation[model]
                case 'custom': label = line_name
                case _:        label = labels[model]

            ps = pred[:, ::stride]
            ps_std = (ps - np.mean(ps)) / np.std(ps)
        
            kde = gaussian_kde(ps_std.flatten())
            plt.plot(x, kde(x), color=color, label=label, alpha=0.7)
    
        plt.title(f'State Distribution Comparison\nD={D}, n_train={n_train}')
        plt.xlabel('Standardized State')
        plt.ylabel('Probability Density')
        plt.yscale('log')
        plt.ylim(1e-4, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()
    # end plot_long_pdfs()

#%%
# HS iteration plots -- GET FROM SIDD
# Read HS file
    def read_hs(self, model, name=None):
        fname = self.name if name is None else name
        filename = f'{fname}_D{self.D:04d}_T{self.n_train:06d}_M{model:02d}.csv'
        df = pd.read_csv(fname)

# =============================================================================
# traj   = result['traj']         # predicted trajectory
# D      = result['D']            # res size
# E      = result['E']            # theta output size (currently always =D)
# n_train= result['n_train']      # training time
# name   = result['name']         # HS_ESN.name --> filename --> name
# model  = result['model']        # index number: ESN 0-7, HS 10-12, truth -1
# truth  = rtruth['traj']         # truth is stored in a separate file
# =============================================================================
