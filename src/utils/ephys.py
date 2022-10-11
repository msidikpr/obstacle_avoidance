import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from src.utils.filter import butter_bandpass
from src.utils.fileops import read_ephys_binary
from src.utils.time import read_time_file

def spikes_from_LFP(ephys_binary_path, ephysT_path, probe_name, spike_thresh=-350, do_timefix=True):
    ephys_offset_val = 0.1
    ephys_drift_rate = -0.000114
    samp_freq = 30000

    if '128' in probe_name:
        n_ch = 128
    elif '64' in probe_name:
        n_ch = 64

    ephys_arr = read_ephys_binary(ephys_binary_path, n_ch, probe_name=probe_name)
    ephys_arr = ephys_arr.to_numpy()

    # highpass filter
    filt_ephys = butter_bandpass(ephys_arr, lowcut=800, highcut=8000, fs=30000, order=6)

    # read in timestamps
    raw_ephysT = pd.DataFrame(read_time_file(ephysT_path))
    # get first/last timepoint, num_samples
    t0 = raw_ephysT.iloc[0,0]
    num_samp = np.size(filt_ephys,0)
    # samples start at t0, and are acquired at rate of n_samples / freq
    ephysT = np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq)

    all_spikeT = []
    for ch in tqdm(range(n_ch)):
        # get the 
        spike_inds = list(np.where(filt_ephys[:,ch] < spike_thresh)[0])
        # get spike times
        spikeT = ephysT[spike_inds]
        if do_timefix:
            # correct the spike times
            spikeT = spikeT - (ephys_offset_val + spikeT * ephys_drift_rate)
        all_spikeT.append(spikeT)

    spikeT_arr = np.array(all_spikeT)

    return spikeT_arr

def calc_PSTH(spikeT, eventT, bandwidth=10, resample_size=1, edgedrop=15, win=1000):
    """
    bandwidth (in msec)
    resample_size (msec)
    edgedrop (msec to drop at the start and end of the window so eliminate artifacts of filtering)
    win = 1000msec before and after
    """
    # some conversions
    bandwidth = bandwidth/1000 # msec to sec
    resample_size = resample_size/1000 # msec to sec
    win = win/1000 # msec to sec
    edgedrop = edgedrop/1000
    edgedrop_ind = int(edgedrop/resample_size)

    # setup time bins
    bins = np.arange(-win-edgedrop, win+edgedrop+resample_size, resample_size)

    # get timestamp of spikes relative to events in eventT
    sps = []
    for i, t in enumerate(eventT):
        sp = spikeT-t
        sp = sp[(sp <= (win+edgedrop)) & (sp >= (-win-edgedrop))] # only keep spikes in this window
        sps.extend(sp)
    sps = np.array(sps) # all values in here are between -1 and 1

    # kernel density estimation
    kernel = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sps[:,np.newaxis])
    density = kernel.score_samples(bins[:,np.newaxis])
    sdf = np.exp(density)*(np.size(sps)/np.size(eventT)) # convert back to spike rate
    sdf = sdf[edgedrop_ind:-edgedrop_ind]

    return sdf

def remove_comp_around_gazeshifts(comp, gazeshift, win=0.25):
    bad_comp = np.array([c for c in comp for g in gazeshift if ((g>(c-win)) & (g<(c+win)))])
    comp_times = np.delete(comp, np.isin(comp, bad_comp))
    return comp_times

def keep_onset_saccades(eventT, win=0.020):
    duplicates = set([])
    for t in eventT:
        new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        duplicates.update(list(new))
    out = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    return out

def keep_arrival_saccades(eventT, win=0.020):
    duplicates = set([])
    for t in eventT:
        new = eventT[((t-eventT)<0.020) & ((t-eventT)>0)]
        duplicates.update(list(new))
    out = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    return out

def calc_sta(self, lag=2, do_rotation=False, using_spike_sorted=True):
        nks = np.shape(self.small_world_vid[0,:,:])
        all_sta = np.zeros([self.n_cells, np.shape(self.small_world_vid)[1], np.shape(self.small_world_vid)[2]])
        plt.subplots(np.ceil(self.n_cells/7).astype('int'), 7, figsize=(35,np.int(np.ceil(self.n_cells/3))), dpi=50)
        if using_spike_sorted:
            cell_inds = self.cells.index
        elif not using_spike_sorted:
            cell_inds = range(self.n_cells)
        for c, ind in enumerate(cell_inds):
            sp = self.model_nsp[c,:].copy()
            sp = np.roll(sp, -lag)
            sta = self.model_vid.T @ sp
            sta = np.reshape(sta, nks)
            nsp = np.sum(sp)
            plt.subplot(np.ceil(self.n_cells/7).astype('int'), 7, c+1)
            ch = int(self.cells.at[ind,'ch'])
            if self.num_channels == 64 or self.num_channels == 128:
                shank = np.floor(ch/32); site = np.mod(ch,32)
            else:
                shank = 0; site = ch
            plt.title(f'ind={ind!s} nsp={nsp!s}\n ch={ch!s} shank={shank!s}\n site={site!s}',fontsize=5)
            plt.axis('off')
            if nsp > 0:
                sta = sta / nsp
                sta = sta - np.mean(sta)
                if do_rotation:
                    sta = np.fliplr(np.flipud(sta))
                plt.imshow(sta, vmin=-0.3 ,vmax=0.3, cmap='seismic')
            else:
                sta = np.nan
                # plt.imshow(np.zeros([120,160]))
            all_sta[c,:,:] = sta
        plt.tight_layout()
        self.sta = all_sta
        if self.figs_in_pdf:
            self.detail_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()
