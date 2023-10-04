"""
File: song.py
Author: Yuki Fujishima
Email: yfujishima1001@gmail.com
Github: https://github.com/yufujis
Description: Library for song analysis
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy import inf, ndarray
from scipy.io import wavfile

from singingmice.utils import integers2slices

logger = logging.getLogger(__name__)


class Song:

    """Class for handling a song"""

    def __init__(self, wf: ndarray, rate: int, singer: int = 1,
                 ref: ndarray = None, fpath: Path = None, load: bool = True):
        """
        Args:
            wf (ndarray): waveform of audio
            rate (int): sampling frequency
            singer (int, optional): singer ID. Defaults to 1.
            ref (ndarray, optional): reference array. Defaults to None.
            fpath (Path, optional): file path. Defaults to None.
            load (bool, optional): weather to load the data. Defaults to True.
        """
        if load:
            self.wf = wf
            self.time: ndarray = np.linspace(
                0, self.wf.shape[0] / rate, self.wf.shape[0]
            )
        self.rate: int = rate
        self.ref: ndarray = ref
        self.singer: int = singer
        self.color: str = f"C{singer-1}"
        self.fpath: Path = fpath
        self.notes: ndarray = np.empty(shape=(1, 2))

    def __repr__(self):
        """ """
        rep = f"Song(singer: {self.singer}, fpath: {self.fpath})"
        return rep

    def add_reference_array(self, ref: ndarray = None):
        """TODO: Docstring for add_reference_array.
        Returns: TODO
        """
        if ref is None:
            ref = self.wf[-self.rate // 2 :]
        self.ref = ref
        logger.info("Added reference")
        return self

    def remove_notes_by_index(self, idxs: List[int]):
        """
        Remove notes by indices. Refind the ones left.

        Args:
            idxs (List[int]): indices of notes to remove
        """
        self.notes = np.delete(self.notes, idxs, axis=0)
        self.find_peakidxs_from_notes()
        self.find_notes_from_peakidxs()
        self.compute_notelens()
        return self

    def compute_wavdecibel(self, window_len: int = 1000, window: str = "hanning"):
        """TODO: Docstring for compute_wavdecibel."""
        self.smo = smooth(abs(self.wf), window_len, window)
        smo_ref = smooth(abs(self.ref), window_len, window)
        self._wave2decibel(noise_arr=smo_ref)
        return self

    def _wave2decibel(
        self,
        noise_win: Tuple[int, int] = None,
        noise_arr: ndarray = None,
        return_result: bool = False,
    ):
        """
        ref (ndarray): reference signal (background noise)
        """
        self.dec = wave2decibel(self.smo, noise_win, noise_arr)
        if return_result:
            return self.dec

        return self


    def compute_notelens(self, return_result: bool = False):
        """Compute note lengths (s) from note positions

        Args:
            return_result (bool, optional): If True, return the ndarray
                            containing the note lengths, otherwise self.
        Returns:
            self.notelens (ndarray): n_notes x 2. note end - note start (s)
        """
        self.notelens = (self.notes[:, 1] - self.notes[:, 0]) / self.rate
        if return_result:
            return self.notelens
        return self

    def stft(
        self,
        nperseg: int = 512,
        window: str = "hann",
        noverlap: int = None,
        return_result: bool = False,
    ):
        """Compute short-time Fourier transform and compute power and
        relative power to the noise.

        Args:
            nperseg (int, optional): n_datapoints per segment. Defaults to 512.
            window (str, optional): type of window. Defaults to "hann".
            noverlap (int, optional): _description_. Defaults to None.
            return_result (bool, optional): If True, returns f, t, z. Defaults to False.

        """
        if noverlap is None:
            noverlap = nperseg // 8
        self.f, self.t, self.z = stft(
            self.wf, self.rate, window, nperseg=nperseg, noverlap=noverlap
        )
        self.power = abs(self.z)
        self.logpower = np.log10(self.power)
        if self.ref is not None:
            f_ref, t_ref, z_ref = stft(
                self.ref, self.rate, window, nperseg=nperseg, noverlap=noverlap
            )
            self.power_ref = abs(z_ref).mean(axis=1, keepdims=True)
            self.decpower = 10 * np.log10(self.power / self.power_ref)
        if return_result:
            return self.f, self.t, self.z
        return self

    def find_peakidxs_from_notes(self):
        """
        Find local maximum indice from notes.
        """
        self.peak_idxs = np.array([np.argmax(self.dec[note[0]:note[1]]) + note[0]
                                  for note in self.notes])
        return self

    def find_notes_from_peakidxs(self):
        """Find notes from peaks.
        """
        cross_thres = define_crossing_thresholds(self.dec, self.peak_idxs)
        self.notes = find_crossings(self.dec, self.peak_idxs, cross_thres)
        return self

    def find_notes(self, peak_thresholds: List[float]):
        """Find notes.

        Args:
            peak_thresholds (List[float]): List of thresholds to find notes.
        Creats:
            self.notes: n_notes x 2. start and end indices for each note
        """
        self.notes = find_notes(self.dec, peak_thresholds)
        return self


def smooth(arr, window_len=1000, window="hanning", mode="same"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        arr: the input signal
        window_len: the dimension of the smoothing window; should be
                    an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming',
                'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array
          instead of a string
    NOTE: length(output) != length(input), to correct this:
          return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[arr[window_len - 1 : 0 : -1], arr, arr[-2 : -window_len - 1 : -1]]

    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    try:
        y = np.convolve(w / w.sum(), s, mode=mode)
    except Exception as e:
        logger.error(e)
    n_remove = arr[window_len - 1 : 0 : -1].shape[0]
    return y[n_remove:-n_remove]


def wave2decibel(
    arr: ndarray, noise_win: Tuple[int, int] = None, noise_arr: ndarray = None
) -> ndarray:
    """convert wave to decibel"""
    if noise_win is None and noise_arr is None:
        raise ValueError("Set either nosie_win or noise_arr.")
    if noise_arr is not None:
        logger.info("Using noise_arr as the noise array...")
        dec = 20 * np.log10(arr / noise_arr.mean())
        dec = 20 * np.log10(arr / noise_arr.mean())
        dec[dec == -inf] = 0
        return dec
    logger.info("Using noise_win to get noise...")
    logger.info("10 * np.log10(arr / arr[noise_win[0]:noise_win[1]].mean())")
    dec = 20 * np.log10(arr / arr[noise_win[0] : noise_win[1]].mean())
    dec[dec == -inf] = 0
    return dec



def read_song(fpath: Path, singer: int = 1, load: bool = True) -> Song:
    """
    Read a file and return a Song object.
    """
    rate, wav = wavfile.read(fpath)
    return Song(wav, rate, singer=singer, fpath=fpath, load=load)


def find_peakidxs_from_slices(dec: ndarray, slice_: List[slice]):
    """Find peak indices from slices of notes.
    Args:
        dec (ndarray): _description_
        slice_ (List[slice]): _description_

    Returns:
        peakidxs (ndarray): n_notes
    """
    peak_idxs: ndarray = np.array([s.start + np.argmax(dec[s.start:s.stop])
                                   for s in slice_])
    return peak_idxs


def find_peakidxs_thres(arr: ndarray, thres: float) -> ndarray:
    slices: List[slice] = integers2slices(np.where(arr > thres)[0])
    peak_idxs: ndarray = find_peakidxs_from_slices(arr, slices)
    return peak_idxs


def find_peakidxs_multi_thres(arr: ndarray, thresholds: List[float]):
    peak_idxs: ndarray = np.hstack([find_peakidxs_thres(arr, thres)
                                    for thres in thresholds])
    peak_idxs = np.unique(peak_idxs.astype(int))
    return peak_idxs


def define_crossing_thresholds(arr: ndarray, peak_idxs: ndarray):
    """Find crossing thresholds for each note peak.
       Assuming values in arr are in logarithmic, it returns 1% of
       the peak (peak - 20) or 1 (bigger of the two).
    Args:
        arr (ndarray): _description_
        peak_idxs (ndarray): _description_
    """
    return np.max([(arr[peak_idxs] - 20, np.ones(peak_idxs.shape[0]))], axis=1).flatten()


def first_cross(arr, start, thres, forward=True):
    if forward:
        where = np.where(arr[start:] <= thres)[0]
        if where.shape[0] == 0:
            return arr.shape[0] - 1
        return min(arr.shape[0] - 1, start + where[0] - 1)
    else:
        try:
            return max(np.where(arr[:start] < thres)[0][-1] + 1, 0)
        except Exception as e:
            return print(e)


def find_single_crossing(arr, start, thres):
    si = first_cross(arr, start, thres, False)
    ei = first_cross(arr, start, thres, True) + 1
    ei = np.min([ei + 1, arr.shape[0]])
    return slice(si, ei)


def find_crossings(dec: ndarray, peak_idxs: ndarray,
                   cross_thresholds: ndarray):
    note_slices = np.array([find_single_crossing(dec, peak_idxs[pos],
                                                 cross_thresholds[pos])
                            for pos in range(len(peak_idxs))])
#   note_slices = np.array([find_single_crossing(dec, idx,
#                                                cross_thresholds[pos])
#                           for pos, idx in enumerate(peak_idxs)])
    notes: ndarray = np.array([[s.start, s.stop] for s in note_slices])
    return notes


def find_notepeaks(dec: ndarray, notes: ndarray):
    """Find peaks from notes

    Args:
        dec (ndarray): Signal to find peaks, typically sound intensity.
        notes (ndarray): n_notes x 2. start and end indices for each note

    Returns:
        maxidxs (ndarray): n_notes. Indices for each peak.
    """
    maxidxs = np.array([np.argmax(dec[note[0]:note[1]]) + note[0]
                        for note in notes])
    return maxidxs


def find_notes(dec: ndarray, peak_thresholds: List[float]):
    """Find notes.

    Args:
        dec (ndarray): signal to find notes, typically sound intensity.
        peak_thresholds (List[float]): List of thresholds to find notes.

    Returns:
        notes: n_notes x 2. start and end indices for each note
    """
    peak_idxs: ndarray = find_peakidxs_multi_thres(dec, peak_thresholds)
    cross_thres: ndarray = define_crossing_thresholds(dec, peak_idxs)
    notes: ndarray = find_crossings(dec, peak_idxs, cross_thres)
    peak_idxs = np.unique(find_notepeaks(dec, notes))
    cross_thres = define_crossing_thresholds(dec, peak_idxs)
    notes = find_crossings(dec, peak_idxs, cross_thres)
    return notes