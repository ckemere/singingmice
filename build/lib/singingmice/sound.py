"""
File: sound.py
Author: Yuki Fujishima
Email: yfujishima1001@gmail.com
Github: https://github.com/yujis
Description: Library for song analysis
"""

import json
import logging
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib import dates as mdates
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from natsort import natsorted
from numpy import inf, ndarray
from numpy.lib.stride_tricks import sliding_window_view
from pandas import DataFrame, Timedelta, Timestamp
from scipy.io import wavfile
from scipy.signal import butter, hilbert, lfilter, resample, stft
from sklearn.linear_model import LinearRegression

from basic import (
    Info,
    avifname_to_datetime,
    integers2slices,
    mkdir,
    read_info,
    read_mouseid_time,
    read_nidaq_dt0,
)

logger = logging.getLogger(__name__)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class Song:

    """Docstring for Song."""

    def __init__(
        self,
        wf: ndarray,
        rate: int,
        singer: int = 1,
        ref: ndarray = None,
        fpath: Path = None,
        info: Info = None,
        load: bool = True,
    ):
        """
        wf (ndarray): waveform
        rate (int): sample rate
        """
        if load:
            self.wf: ndarray = wf
            self.time: ndarray = np.linspace(
                0, self.wf.shape[0] / rate, self.wf.shape[0]
            )
        self.rate: int = rate
        self.ref: ndarray = ref
        self.singer: int = singer
        self.color: str = f"C{singer-1}"
        self.fpath: Path = fpath
        #       self.fpath = info.takedir / "song" / f"M{mouse_id1}-{song_id}_avi.wav"
        #       self._notesfpath = info.takedir / "song" / f"M{mouse_id1}_avi_notes-{song_id}.wav"
        if info is not None:
            self.info = info
        self.fhead = str(self.fpath).split("\\")[-1].split(".")[0]
        self.dirpath = Path(str(self.fpath).split(self.fhead)[0])
        self.decfpath = self.dirpath / f"{self.fhead}_dec.npy"
        self.fig: Figure = None
        self.imgfpath: Path = None
        self.start_ts: Timestamp = None
        # self.dt =

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

    def add_info(self, take_name: str):
        """
        Add session information
        """
        self.info: Info = read_info(take_name)
        return self

    def read_reference_file(self, fpath: Path = None):
        """TODO: Docstring for read_reference_file.
        Returns: TODO
        """
        if fpath is None:
            fpath = self.info.takedir / f"{self.info.take_name}_ref.wav"
        rate, ref = wavfile.read(fpath)
        if rate != self.rate:
            logger.error(
                "Sample rate does not match - wf: %s, ref: %s", self.rate, rate
            )
        self.ref = ref
        logger.info("Loaded %s as reference", fpath)
        return self

    def determine_crossthresholds(self):
        """TODO: Docstring for determine_crossthresholds.
        Returns: TODO
        """
        cross_thres = np.max(
            [(self.dec[self.maxidxs] - 20, np.ones(self.maxidxs.shape[0]))], axis=1
        ).flatten()
        self.cross_thres = cross_thres
        return self

    def _find_crossing(self, arr, start, thres):
        si = self._first_cross(arr, start, thres, False)
        ei = self._first_cross(arr, start, thres, True) + 1
        ei = min(ei + 1, arr.shape[0])
        return slice(si, ei)

    def _first_cross(self, arr, start, thres, forward=True):
        if forward:
            where = np.where(arr[start:] <= thres)[0]
            if where.shape[0] == 0:
                return arr.shape[0] - 1
            return min(arr.shape[0] - 1, start + where[0] - 1)
        else:
            try:
                return max(np.where(arr[:start] < thres)[0][-1] + 1, 0)
            except Exception as e:
                logger.error(e)
                return 0

    def find_notes_from_slices(self):
        """TODO: Docstring for find_notes_from_slices.
        Returns: TODO
        """
        self.notes = np.array([[s.start, s.stop] for s in self.slices])
        return self

    def find_notes_from_crossthresholds(self):
        """TODO: Docstring for find_notes_from_crossthresholds.
        Returns: TODO
        """
        slices = np.array(
            [
                self._find_crossing(self.dec, self.maxidxs[i], self.cross_thres[i])
                for i in range(len(self.cross_thres))
            ]
        )
        self.slices = slices
        self.find_notes_from_slices()
        return self

    def read_audio_reference_file(self, fpath: Path = None):
        """TODO: Docstring for read_reference_file.
        Returns: TODO
        """
        if fpath is None:
            fpath = self.info.takedir / f"{self.info.take_name}_ref.npy"
        ref = np.load(fpath)
        self.ref = ref
        logger.info("Loaded %s as reference", fpath)
        return self

    def add_start_i(self, start_i: int):
        """
        add start_i. Use when making a new song with grab_audiochunk2.

        """
        self.start_i: int = start_i
        return self

    def add_notes(self, notes: ndarray):
        """TODO: Docstring for add_note_idxs.
        Returns: TODO

        """
        self.notes = notes
        self.length = (self.notes[-1][1] - self.notes[0][0]) / self.rate
        self.compute_notelens()
        logger.info(
            "Added notes: # %s, song length %s (s)", self.notes.shape[0], self.length
        )
        return self

    def resample(self, orig_rate: int = 10000, new_rate: int = 250000):
        """TODO: Docstring for resample.
        Resample to avisoft's sample rate.
        """
        self.rate = new_rate
        self.wf = resample(self.wf, int(self.wf.shape[0] * new_rate / orig_rate))
        return self

    def normalize_wf(self):
        """TODO: Docstring for normalize_wf."""
        self.norm_wf = self.wf / np.max(self.wf)
        return self

    def add_starttime(self, ts: Timestamp):
        self.start_ts = ts
        self._compute_onset_offset_with_start_ts()
        return self

    def add_starttime_from_avifname(self, f: bool, head_num: int = 0):
        fname = str(self.fpath).split("\\")[-1]
        self.start_ts = avifname_to_datetime(fname, head_num, f)
        start = self.start_ts
        end = self.start_ts + Timedelta(seconds=self.wf.shape[0] / self.rate)
        self.dt = pd.date_range(start=start, end=end, periods=self.wf.shape[0])
        self._compute_onset_offset_with_start_ts()
        return self

    def add_starttime_from_start_i(self):
        self.start_ts = self.info.nidaq_dt0 + Timedelta(
            seconds=self.start_i / self.rate
        )
        self._compute_onset_offset_with_start_ts()
        return self

    def add_dts_from_start_ts(self):
        """TODO: Docstring for add_dts_from_start_ts."""
        end_ts = self.start_ts + Timedelta(self.wf.shape[0] / self.rate, unit="s")
        self.dt = pd.date_range(
            start=self.start_ts, end=end_ts, periods=self.wf.shape[0]
        )
        return self

    def _compute_onset_offset_with_start_ts(self):
        """"""
        try:
            ondelta = Timedelta(seconds=self.notes[0][0] / self.rate)
            self.onset_ts: Timestamp = self.start_ts + ondelta
            offdelta = Timedelta(seconds=self.notes[-1][1] / self.rate)
            self.offset_ts: Timestamp = self.start_ts + offdelta
        except Exception as e:
            logger.warning("No notes. Onset and offset are not set.")
        return self

    def read_notes_file(self, fpath: Path = None):
        """TODO: Docstring for read_notes_file."""
        if fpath is None:
            first = str(self.fpath).split("_avi.")[0]
            head, num = first.split("\\")[-1].split("-")
            #           head = str(self.fpath).split(".")[0]
            #           num = str(self.fpath).split(".")[0].split("-")[1].split("_")[0]
            dirpath = Path(str(self.fpath).split(head)[0])
            fpath = dirpath / str(head + "_avi_notes-" + num + ".npy")
        logger.info("Loading %s", fpath)
        self._notesfpath = fpath
        self.notesfpath = fpath
        self.notes = np.load(fpath)
        self.length = (self.notes[-1][1] - self.notes[0][0]) / self.rate
        self.compute_notelens()
        logger.info(
            "Added notes: # %s, song length %s (s)", self.notes.shape[0], self.length
        )
        return self

    def read_avi_maxidx(self, fpath: Path = None):
        """TODO: Docstring for read_avi_maxidx."""
        if fpath is None:
            first = str(self.fpath).split("_avi.")[0]
            head, num = first.split("\\")[-1].split("-")
            #           head = str(self.fpath).split(".")[0]
            #           num = str(self.fpath).split(".")[0].split("-")[1].split("_")[0]
            dirpath = Path(str(self.fpath).split(head)[0])
            fpath = dirpath / str(head + "_avi_maxidx-" + num + ".npy")
        self.maxidx_fpath = fpath
        logger.info("Loading %s", self.maxidx_fpath)
        self.maxidxs = np.load(self.maxidx_fpath)
        return self

    def save_wf_wav(self, fpath: Path = None):
        """ """
        if fpath is None:
            if self.fpath is not None:
                fpath = self.fpath
            else:
                dthead: str = self.start_ts.strftime("%Y-%m-%d_%H-%M-%S_%f")
                self.fhead: str = f"{dthead}_M{self.singer}"
                fname: str = f"{self.fhead}.wav"
                self.avidir: Path = mkdir(self.info.takedir / "avisong")
                self.avimousedir: Path = mkdir(self.avidir / f"M{self.singer}")
                self.fpath = self.avimousedir / fname
        wavfile.write(fpath, self.rate, self.wf)
        logger.info("Saved: %s", fpath)
        return self

    def find_fhead(self):
        """TODO: Docstring for find_fhead.
        Returns: TODO

        """
        self.fhead = str(self.fpath).split("\\")[-1].split(".")[0]
        return self

    def save_dec(self, fpath: Path = None):
        """TODO: Docstring for save_dec."""
        if fpath is None:
            if hasattr(self, "decfpath"):
                fpath = self.decfpath
            else:
                fname = f"{self.fhead}_dec.npy"
                fpath = self.dirpath / fname

        np.save(fpath, self.dec)
        logger.info("Saved: %s", fpath)
        return self

    def read_dec(self, fpath: Path = None):
        """TODO: Docstring for read_dec."""
        if fpath is None:
            if hasattr(self, "decfpath"):
                self.dec = np.load(self.decfpath)
            else:
                fname = f"{self.fhead}_dec.npy"
                self.decfpath = self.dirpath / fname
        self.dec = np.load(self.decfpath)
        logger.info("Loaded: %s", self.decfpath)
        return self

    def save_avinotes_file(self, fpath: Path = None, maxidx_fpath: Path = None):
        """TODO: Docstring for save_notes_file.
        Returns: TODO

        """
        if fpath is None:
            if hasattr(self, "_notesfpath"):
                fpath = self._notesfpath
            else:
                dthead: str = self.start_ts.strftime("%Y-%m-%d_%H-%M-%S_%f")
                self.fhead = f"{dthead}_M{self.singer}"
                fname = f"{self.fhead}_notes.npy"
                self.avidir = mkdir(self.info.takedir / "avisong")
                self.avimousedir = mkdir(self.avidir / f"M{self.singer}")
                self._notesfpath = self.avimousedir / fname
                fpath = self._notesfpath

        np.save(fpath, self.notes)
        logger.info("Saved: %s", fpath)
        if maxidx_fpath is None:
            if hasattr(self, "maxidx_fpath"):
                maxidx_fpath = self.maxidx_fpath
            else:
                maxidx_fpath = (
                    fname.split("notes")[0] + "maxidx" + fname.split("notes")[-1]
                )
        np.save(maxidx_fpath, self.maxidxs)
        logger.info("Saved: %s", maxidx_fpath)
        return self

    def save_avimaxidxs(self, fpath: Path = None):
        """TODO: Docstring for save_avimaxidxs."""
        if fpath is None:
            if hasattr(self, "_notesfpath"):
                fpath = self._notesfpath
            else:
                dthead: str = self.start_ts.strftime("%Y-%m-%d_%H-%M-%S_%f")
                self.fhead = f"{dthead}_M{self.singer}"
                fname = f"{self.fhead}_notes.npy"
                self.avidir = mkdir(self.info.takedir / "avisong")
                self.avimousedir = mkdir(self.avidir / f"M{self.singer}")
                self._notesfpath = self.avimousedir / fname
                fpath = self._notesfpath

        np.save(fpath, self.maxidx_fpath)
        logger.info("Saved: %s", fpath)
        return self

    def read_audionotes_file(
        self, fpath: Path = None, start_fpath: Path = None, maxidx_fpath: Path = None
    ):
        """read notes of audiotech files"""
        if fpath is None:
            num: str = str(self.fpath).split(".")[0].split("-")[-1]
            fhead: str = str(self.fpath).split("\\")[-1].split("-")[-2]
            fpath = self.info.takedir / "song" / f"{fhead}_audio_notes-{num}.npy"
            self._notesfpath = fpath
            self._notesfpath = fpath
        if start_fpath is None:
            num = str(self.fpath).split(".")[0].split("-")[-1]
            fhead: str = str(self.fpath).split("\\")[-1].split("-")[-2]
            start_fpath = self.info.takedir / "song" / f"{fhead}_start-{num}.npy"
        if maxidx_fpath is None:
            num = str(self.fpath).split(".")[0].split("-")[-1]
            fhead: str = str(self.fpath).split("\\")[-1].split("-")[-2]
            self.maxidx_fpath = self.info.takedir / "song" / f"{fhead}_maxidx-{num}.npy"
            maxidx_fpath = self.maxidx_fpath
        try:
            logger.info("Loading %s", fpath)
            self.notes = np.load(fpath)
        except Exception as e:
            logger.warning(e)
        try:
            logger.info("Loading %s", maxidx_fpath)
            self.maxidxs = np.load(maxidx_fpath)
        except Exception as e:
            logger.warning(e)
        try:
            logger.info("Loading %s", maxidx_fpath)
            self.maxidxs = np.load(maxidx_fpath)
        except Exception as e:
            logger.warning(e)
        try:
            logger.info("Loading %s", start_fpath)
            self.start_i = np.load(start_fpath)[0]
            self.notes -= self.start_i
        except Exception as e:
            logger.warning(e)
        try:
            self.length = (self.notes[-1][1] - self.notes[0][0]) / self.rate
        except Exception as e:
            logger.warning(e)
        try:
            self.compute_notelens()
        except Exception as e:
            logger.warning(e)
        try:
            logger.info(
                "Added notes: # %s, song length %s (s)",
                self.notes.shape[0],
                self.length,
            )
        except Exception as e:
            logger.warning(e)
        return self

    def read_start_i(self, start_fpath: str = None):
        """
        Read start_i
        """
        if start_fpath is None:
            num = str(self.fpath).split(".")[0].split("-")[-1]
            fhead: str = str(self.fpath).split("\\")[-1].split("-")[-2]
            start_fpath = self.info.takedir / "song" / f"{fhead}_start-{num}.npy"
        try:
            logger.info("Loading %s", start_fpath)
            self.start_i = np.load(start_fpath)[0]
            self.notes -= self.start_i
        except Exception as e:
            logger.warning(e)
        return self

    def compute_bandpower(
        self,
        lowcut: int,
        highcut: int,
        fs: int = 250000,
        order: int = 5,
        smooth_window: int = None,
    ):
        """
        Compute band power using a butterworth filter
        and hilbert transform
        """
        self.filtered_wf = butter_bandpass_filter(self.wf, lowcut, highcut, fs, order)
        self.hilbert_wf = hilbert(self.filtered_wf)
        self.filtered_ref = butter_bandpass_filter(self.ref, lowcut, highcut, fs, order)
        self.hilbert_ref = hilbert(self.filtered_ref)
        if smooth_window is not None:
            self.bandpw = smooth(np.abs(self.hilbert_wf), smooth_window)
            self.bandpw_ref = smooth(np.abs(self.hilbert_ref), smooth_window)
            self.decband = np.log10(self.bandpw / self.bandpw_ref.mean()) * 10
        else:
            self.bandpw = np.abs(self.hilbert_wf)
            self.bandpw_ref = np.abs(self.hilbert_wf)
        return self

    def find_notes_from_bandpower(
        self, bandpw_thres: float, cross_thres: float, timegap: float = 0.5
    ):
        """ """
        above_thres = np.where(self.decband >= bandpw_thres)[0]
        self.slices = np.array(integers2slices(above_thres))
        self.maxidxs = np.array(
            [np.argmax(self.dec[s.start : s.stop]) + s.start for s in self.slices]
        )
        self.keep_bigger_notegroup(timegap)
        self.find_noteidxs(cross_thres)
        for i, note in enumerate(self.notes):
            if note[0] == note[1]:
                self.notes[i, 0] -= 1
                self.notes[i, 1] += 1
        return self

    def save_wf(self, fpath: Path = None):
        """ """
        if fpath is None:
            fpath = self.fpath
        try:
            np.save(fpath, self.wf)
            logger.info("Saved: %s", fpath)
        except Exception as e:
            logger.error(e)
        return self

    def save_wav(self, fpath: Path = None):
        """TODO: Docstring for save_wav."""
        if fpath is None:
            fpath = self.fpath
        try:
            wavfile.read(fpath)
            logger.info("Saved: %s", fpath)
        except Exception as e:
            logger.error(e)
        return self

    def save_audionotes(
        self, fpath: Path = None, start_fpath: Path = None, maxidx_fpath: Path = None
    ):
        """save notes of audiotech files"""
        if fpath is None:
            num: str = str(self.fpath).split(".")[0].split("-")[-1]
            fhead: str = str(self.fpath).split("\\")[-1].split("-")[-2]
            fpath = self.info.takedir / "song" / f"{fhead}_audio_notes-{num}.npy"
        if start_fpath is None:
            num = str(self.fpath).split(".")[0].split("-")[-1]
            fhead: str = str(self.fpath).split("\\")[-1].split("-")[-2]
            start_fpath = self.info.takedir / "song" / f"{fhead}_start-{num}.npy"
        if maxidx_fpath is None:
            num = str(self.fpath).split(".")[0].split("-")[-1]
            fhead: str = str(self.fpath).split("\\")[-1].split("-")[-2]
            maxidx_fpath = self.info.takedir / "song" / f"{fhead}_maxidx-{num}.npy"
        np.save(fpath, self.notes + self.start_i)
        logger.info("Saved %s", fpath)
        np.save(start_fpath, np.array([self.start_i]))
        logger.info("Saved %s", start_fpath)
        try:
            np.save(maxidx_fpath, self.maxidxs)
            logger.info("Saved %s", maxidx_fpath)
        except Exception as e:
            logger.info(e)
        return self

    def save_audiomaxidx(self, fpath: Path = None, start_fpath: Path = None):
        """save notes of audiotech files"""
        if fpath is None:
            num: str = str(self.fpath).split(".")[0].split("-")[-1]
            fhead: str = str(self.fpath).split("\\")[-1].split("-")[-2]
            fpath = self.info.takedir / "song" / f"{fhead}_audio_maxidx-{num}.npy"
        np.save(fpath, self.maxidxs + self.start_i)
        logger.info("Saved %s", fpath)
        return self

    def plot_wf_dec(self, kind: str, show_notes: bool = True):
        """TODO: Docstring for plot_wf_dec.
        Returns: TODO
        """
        fig, ax = plt.subplots(1, tight_layout=True)
        if kind == "audio":
            ts = self.info.nidaq_dt0 + Timedelta(seconds=self.start_i / self.rate)
        else:
            ts = ""
        try:
            title: str = f"{self.fhead} {ts}, notes: {self.notes.shape[0]}"
            ax.set_title(title)
        except Exception as e:
            logger.warning(e)
        ax.plot(self.time, self.wf, color=self.color)
        ax.set_facecolor("black")
        ax_1 = ax.twinx()
        ax_1.plot(self.time, self.dec, color="white", alpha=0.8)
        ax_1.set_ylim(0, 50)
        if show_notes:
            ax_1.vlines(
                self.time[self.notes[:, 0]], ymin=0, ymax=30, color="white", alpha=0.6
            )
            ax_1.vlines(
                self.time[self.notes[:, 1]], ymin=0, ymax=30, color="pink", alpha=0.6
            )
        self.fig = fig
        return self.fig

    def _plot_wf(self, ax: Axes, show_notes: bool, decimate: bool):
        """TODO: Docstring for _plot_wf."""
        ax_1 = ax.twinx()
        if decimate:
            ax.plot(self.time[::10], self.wf[::10], color=self.color)
            ax_1.plot(self.time[::10], self.dec[::10], color="white", alpha=0.8)
        else:
            ax.plot(self.time, self.wf, color=self.color)
            ax_1.plot(self.time, self.dec, color="white", alpha=0.8)
        ax.set_facecolor("black")
        ax_1.set_ylim(-5, 50)
        if show_notes:
            ax_1.vlines(
                self.time[self.notes[:, 0]], ymin=0, ymax=20, color="white", alpha=0.6
            )
            ax_1.vlines(
                self.time[self.notes[:, 1]], ymin=0, ymax=20, color="pink", alpha=0.6
            )
            ax_1.scatter(self.time[self.maxidxs], self.dec[self.maxidxs], color="red")
            [
                ax_1.annotate(
                    text=pos, xy=(self.time[maxidx], self.dec[maxidx]), color="orange"
                )
                for pos, maxidx in enumerate(self.maxidxs)
            ]
        return ax, ax_1

    def _plot_notelens_maxintv(self, ax: Axes):
        """TODO: Docstring for _plot_notes."""
        ax.plot(
            self.notelens[:, 0], self.notelens[:, 1], color="black", label="Note Length"
        )
        ax.set_ylabel("Time Length (s)")
        ax.plot(
            self.time[self.maxidxs[1:]],
            self.maxintv / self.rate,
            color="red",
            label="Max Interval",
        )
        ax.set_ylim(0, 0.2)
        ax.legend()
        return ax

    def _plot_maxintensity(self, ax: Axes):
        """TODO: Docstring for _plot_maxintv."""
        ax.scatter(
            self.time[self.maxidxs],
            self.dec[self.maxidxs],
            color="red",
            label="Maxmal Intensity",
        )
        ax.set_ylabel("Maxmal Intensity (dB)", color="red")
        ax.set_ylim(self.dec[self.maxidxs].min(), self.dec[self.maxidxs].max())
        [
            ax.annotate(
                text=pos, xy=(self.time[maxidx], self.dec[maxidx]), color="black"
            )
            for pos, maxidx in enumerate(self.maxidxs)
        ]
        ax.legend()
        return ax

    def _show_tf(self, ax: Axes, show_notes: bool):
        """TODO: Docstring for _show_tf.
        Returns: TODO

        """
        ax.imshow(
            self.decpower,
            origin="lower",
            aspect="auto",
            cmap="jet",
            extent=[self.t[0], self.t[-1], self.f[0], self.f[-1]],
            vmin=0,
        )
        if show_notes:
            ax.vlines([self.time[self.maxidxs]], 0, 10000, color="white")
        return ax

    def _plot(
        self,
        kind: str,
        show_notes: bool = True,
        show_tf: bool = False,
        decimate: bool = True,
        show_decband: bool = False,
    ):
        """TODO: Docstring for plot."""
        if show_tf:
            fig, axs = plt.subplots(
                4,
                figsize=(12, 8),
                sharex=True,
                tight_layout=True,
                gridspec_kw={"height_ratios": [3, 3, 1, 1]},
            )
        else:
            fig, axs = plt.subplots(
                3,
                figsize=(12, 6),
                sharex=True,
                tight_layout=True,
                gridspec_kw={"height_ratios": [3, 1, 1]},
            )
        ax = axs[0]
        if kind == "audio":
            ts = self.info.nidaq_dt0 + Timedelta(seconds=self.start_i / self.rate)
        else:
            ts = ""
        try:
            title: str = f"{self.fhead} {ts}, notes: {self.notes.shape[0]}"
            ax.set_title(title)
        except Exception as e:
            logger.warning(e)
        ax, ax_1 = self._plot_wf(axs[0], show_notes, decimate)
        if show_decband:
            ax_1.plot(self.time, self.decband, color="green", alpha=0.8)
        if show_tf:
            axs[1] = self._show_tf(axs[1], show_notes)
        if show_notes:
            axs[-2] = self._plot_notelens_maxintv(axs[-2])
            axs[-1] = self._plot_maxintensity(axs[-1])
        self.fig = fig
        return self.fig

    def plot(self, tf: bool = True):
        """TODO: Docstring for plot."""
        fig, axs = plt.subplots(
            2,
            figsize=(12, 8),
            sharex=True,
            tight_layout=True,
        )
        self.fig = fig
        axs[0].plot(self.dt, self.wf, color="black")
        axs[0].set_ylim(-1, 1)
        xlims = mdates.date2num([self.dt[0], self.dt[-1]])
        try:
            axs[1].imshow(
                self.decpower,
                origin="lower",
                aspect="auto",
                cmap="turbo",
                vmin=0,
                extent=[xlims[0], xlims[-1], self.f[0], self.f[-1]],
            )
        except:
            pass
        ax1 = axs[1].twinx()
        ax1.set_ylim(0, 0.2)
        try:
            ax1.scatter(
                self.dt[self.decmaxidxs], self.notelens[:, 1], color="lightgreen"
            )
        except Exception as e:
            raise e

        ax2 = ax1.twinx()
        ax2.plot(self.dt, self.dec, color="pink", alpha=0.5)
        try:
            ax2.scatter(self.dt[self.decmaxidxs], self.decmaxs, color="red")
            [
                ax2.annotate(ii, (self.dt[maxi], self.decmaxs[ii]), color="white")
                for ii, maxi in enumerate(self.decmaxidxs)
            ]
        except Exception as e:
            raise e
        axs[0].set_title(f"{self.fhead}")

        return fig

    def plot2(self, tf: bool = True):
        """TODO: Docstring for plot."""
        fig, axs = plt.subplots(
            2,
            figsize=(10, 4),
            sharex=True,
            tight_layout=True,
            gridspec_kw={"height_ratios": [1, 2]},
        )
        axs[0].plot(self.time, self.wf, color="black", linewidth=0.5)
        axs[0].set_ylim(-1, 1)
        if tf:
            axs[1].imshow(
                self.decpower,
                origin="lower",
                aspect="auto",
                cmap="turbo",
                vmin=0,
                extent=[self.t[0], self.t[-1], self.f[0], self.f[-1]],
            )
            axs[1].tick_params(labelleft=False)
        ax1 = axs[1].twinx()
        ax1.set_ylim(0, 0.2)
        try:
            ax1.scatter(
                self.time[self.decmaxidxs], self.notelens[:, 1], color="k", s=10
            )
        except Exception as e:
            raise e

        ax2 = ax1.twinx()
        ax2.plot(self.time, self.dec, color="pink", alpha=0.5, linewidth=0.5)
        try:
            ax2.scatter(self.time[self.decmaxidxs], self.decmaxs, color="red", s=10)
            for ii, maxi in enumerate(self.decmaxidxs):
                ax2.annotate(
                    ii, (self.time[maxi], self.decmaxs[ii]), color="white", fontsize=8
                )
        except Exception as e:
            raise e
        axs[0].set_title(f"{self.fhead}")
        self.fig = fig
        return fig

    def remove_notes_by_index(
        self,
        idxs: List[int],
        refind: bool = False,
        cross_thres: float = 1,
        align: str = None,
    ):
        """TODO: Docstring for remove_notes."""
        self.notes = np.delete(self.notes, idxs, axis=0)
        self.maxidxs = np.delete(self.maxidxs, idxs, axis=0)
        if refind:
            self.find_noteidxs(cross_thres)
            for i, note in enumerate(self.notes):
                if note[0] == note[1]:
                    self.notes[i, 0] -= 1
                    self.notes[i, 1] += 1
        self.compute_notelens()
        self.find_maxinterval()
        if align is not None:
            self.align_time_to(align)
        return self

    def remove_notes(self, idxs: List[int], align: str = None):
        """TODO: Docstring for remove_notes."""
        self.notes = np.delete(self.notes, idxs, axis=0)
        self.maxidxs = np.delete(self.maxidxs, idxs, axis=0)
        self.compute_notelens()
        self.find_maxinterval()
        if align is not None:
            self.align_time_to(align)
        return self

    def savefig(self, savepath: Path = None, fig: Figure = None):
        """
        Save a figure
        """
        if savepath is None:
            savepath = str(self.fpath).split(".")[0] + ".jpg"
        if fig is None:
            fig = self.fig
        fig.savefig(savepath)
        logger.info("Saved: %s", savepath)
        return self

    def smooth(self, window_len: int, window: str, return_result: bool = False):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with
        the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are
        minimized in the begining and end part of the output signal.

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
        self.smo: ndarray = smooth(abs(self.wf), window_len, window)
        if return_result:
            return self.smo
        return self

    def compute_wavdecibel(self, window_len: int = 1000, window: str = "hanning"):
        """TODO: Docstring for compute_wavdecibel."""
        self.smooth(window_len, window)
        smo_ref = smooth(abs(self.ref), window_len, window)
        self.wave2decibel(noise_arr=smo_ref)
        return self

    def compute_note_intensity(self):
        """TODO: Docstring for compute_note_intensity."""
        note_max = np.zeros(self.notes.shape)
        for pos, note in enumerate(self.notes):
            note_max[pos, 0] = note[0] + np.argmax(self.dec[note[0] : note[1]])
            note_max[pos, 1] = max(self.dec[note[0] : note[1]])
        self.note_max: ndarray = note_max
        return self

    def wave2decibel(
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

    def align_time_to(self, align_to: str = "onset"):
        """TODO: Docstring for align_time_to.
        Returns: TODO
        """
        if align_to.lower() == "onset":
            align_time = self.time[int(self.notes[0][0])]
        elif align_to.lower() == "offset":
            align_time = self.time[int(self.notes[-1][1])]
        self.time -= align_time
        try:
            self.t -= align_time
        except:
            pass
        try:
            self.notelens[:, 0] -= align_time
        except:
            pass
        #       try:
        #           self.notelens[:, 0] -= align_time
        #       except Exception as e:
        #           raise e
        return self

    def compute_notelens(self, return_result: bool = False):
        """TODO: Docstring for compute_notelen.
        Returns: TODO
        """
        self.notelens = np.zeros(self.notes.shape, dtype=float)
        self.notelens[:, 0] = self.time[self.notes[:, 0]]
        self.notelens[:, 0] = self.time[self.notes[:, 0]]
        self.notelens[:, 1] = self.notes[:, 1] - self.notes[:, 0]
        self.notelens[:, 1] /= self.rate
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
        """TODO: Docstring for stft.
        Returns: TODO
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

    def _find_edges(self, arr: ndarray, win: ndarray, local_thres: float):
        """
        Find local minimum in an array window
        """
        return np.where(arr > min(arr) + local_thres) + win[0]

    def find_localedges(
        self, window_size: int = 2000, step: int = 100, local_thres: float = 5.0
    ):
        """Docstring for find_localedges.
        Find datapoints that exceed the minimum value within the local window.
        This should account for the backgrund noise where the absolute
        amplitude stays high during vocalization.
        """
        win_idxs = sliding_window_view(np.arange(0, self.dec.shape[0], 1), window_size)[
            ::step
        ]
        arrs = sliding_window_view(self.dec, window_size)[::step]
        edges = [
            self._find_edges(arr, win_idxs[i], local_thres)
            for i, arr in enumerate(arrs)
        ]
        # concatenate all edges found in individual windows
        # delete overlapping indices.
        self.edges = np.unique(np.squeeze(np.concatenate(edges, axis=1)))
        logger.info("Found %s unique edges.", len(edges))
        return self

    def localedge_slices(
        self, window_size: int = 2000, step: int = 100, local_thres: float = 5.0
    ):
        """
        Find local edges and make index slices.
        """
        self.find_localedges(window_size, step, local_thres)
        self.slices = np.array(integers2slices(self.edges))
        return self

    def find_maxidxs(self):
        """
        Find local maximum indice.
        """
        self.maxidxs = np.array(
            [s.start + np.argmax(self.dec[s.start : s.stop]) for s in self.slices]
        )
        return self

    def find_decmax(self):
        """TODO: Docstring for find_decmax.
        Returns: TODO
        """
        self.decmaxidxs = np.array(
            [n[0] + np.argmax(self.dec[n[0] : n[1]]) for n in self.notes]
        )
        self.decmaxs = np.array([np.max(self.dec[n[0] : n[1]]) for n in self.notes])
        return self

    def find_maxidxs_from_notes(self):
        """
        Find local maximum indice from notes.
        """
        self.maxidxs = np.array(
            [np.argmax(self.dec[note[0] : note[1]]) + note[0] for note in self.notes]
        )
        return self

    def find_maxinterval(self):
        """ """
        self.maxintv = np.array(
            [self.maxidxs[i + 1] - idx for i, idx in enumerate(self.maxidxs[:-1])]
        )
        return self

    def cluster_notecandidates(self, timegap: float = 0.6):
        """TODO: Docstring for cluster_notecandidates.
        self.localedge_slices(window_size, step, local_thres)
        Group potential notes by gaps between the local maximums.
        If the local maximum occures after > timegaps (s), it will
        be treated as a separate song (candidate).
        * Returning indices of max indices
        """
        maxidxs_groups = []
        st_ = self.time[self.maxidxs][0]
        group = [0]
        for i, st in enumerate(self.time[self.maxidxs][1:], 1):
            if st - st_ > timegap:
                maxidxs_groups.append(group)
                group = []
            st_ = st
            group.append(i)
        maxidxs_groups.append(group)
        return maxidxs_groups

    def remove_outlier(self, arg1):
        """TODO: Docstring for remove_outlier.

        Args:
            arg1 (TODO): TODO

        Returns: TODO

        """

    def keep_bigger_notegroup(self, timegap: float = 0.5):
        """TODO: Docstring for keep_bigger_notegroup."""
        groups = self.cluster_notecandidates(timegap)
        bigger_idxs = groups[np.argmax([len(group) for group in groups])]
        logger.info(
            "N of potential notes in each group %s. Keep the biggest",
            [len(group) for group in groups],
        )
        self.maxidxs = self.maxidxs[bigger_idxs]
        if hasattr(self, "slices"):
            self.slices = self.slices[bigger_idxs]
            self.find_notes_from_slices()
        else:
            print("no slices")
            if hasattr(self, "notes"):
                self.notes = self.notes[bigger_idxs, :]
                self.slices = np.array(slice(note[0], note[1]) for note in self.notes)
                print("created slices from notes")
            else:
                print("no notes")
        return self

    def find_noteidxs(self, cross_thres: float):
        """TODO: Docstring for find_noteidxs."""
        logger.info("Will look for start and end of %s notes.", len(self.maxidxs))
        notes = []
        for i, maxidx in enumerate(self.maxidxs):
            if i == 0:
                win_start = max(0, maxidx - (self.maxidxs[1] - maxidx))
            else:
                win_start = max(0, self.maxidxs[i - 1])
            if i == len(self.maxidxs) - 1:
                win_end = min(self.dec.shape[0], maxidx + (maxidx - self.maxidxs[-2]))
            else:
                win_end = min(self.dec.shape[0], self.maxidxs[i + 1])
            win = Window(self.dec, (win_start, win_end))
            # Find start
            try:
                start = win.first_cross(
                    1, increase=True, start_i=maxidx - win.win_idx[0]
                )
            except Exception as e:
                #               logger.info(e)
                try:
                    argmin = np.argmin(win.data[: maxidx - win.win_idx[0]])
                    start = argmin + win.win_idx[0]
                except Exception as e:
                    logger.info(e)
                    continue
            # Find end
            try:
                end = win.first_cross(
                    1, increase=False, start_i=maxidx - win.win_idx[0]
                )
            except Exception as e:
                #               logger.info(e)
                end = np.argmin(win.data[maxidx - win.win_idx[0] :]) + maxidx
            notes.append([start, end])
        self.notes = np.array(notes)
        self.notes = np.unique(self.notes, axis=0)
        return self

    def detect_notes(
        self,
        window_size: int = 2000,
        step: int = 100,
        local_thres: float = 5.0,
        timegap: float = 0.6,
        cross_thres: float = 1.0,
    ):
        """
        Find local edges, group them, find maximum of each group.
        """
        self.localedge_slices(window_size, step, local_thres)
        self.find_maxidxs()
        self.keep_bigger_notegroup(timegap)
        self.find_noteidxs(cross_thres)
        for i, note in enumerate(self.notes):
            if note[0] == note[1]:
                self.notes[i, 0] -= 1
                self.notes[i, 1] += 1
        if self.notes[-1, 1] == self.wf.shape[0]:
            self.notes = self.notes[:-1, :]
        return self

    def _find_notestart(self, note_idx: int):
        """TODO: Docstring for _find_notestart.
        Returns: TODO

        """
        win = Window(self.dec, ())

    def _detect_notes2(
        self,
        thres: float = 5.0,
        cross_thres: float = 5.0,
        n_dp: int = 4000,
        detect_win: int = 25000 + 12500,
        return_result: bool = False,
    ) -> ndarray:
        """
        Detect notes using linear regression.
        Returns
            note_idxs (ndarray): indices (within the song array) of the start
                                 and end of all notes. Shape: (n_notes, 2)
        """
        logger.info(
            "thres: %s  cross_thres: %s  n_dp: %s  detect_win: %s",
            thres,
            cross_thres,
            n_dp,
            detect_win,
        )

        dec = self.dec

        start_idxs: List[int] = []
        end_idxs: List[int] = []

        #   #   n_dp = 4000
        #   #   detect_win: int = 25000 + 12500

        idx = 0
        count = 0
        while True:
            if idx + n_dp <= dec.shape[0]:
                chunk = Window(dec, win_idx=(idx, idx + n_dp))
            else:
                chunk = Window(dec, win_idx=(idx, dec.shape[0]))
            if chunk.max_smaller_than(thres):
                # go next
                idx += n_dp
            else:
                chunks = []
                while True:
                    chunks.append(chunk)
                    # perform linear regression on the chunk
                    # if coef > 0, the slope is still going up
                    # when coef < 0, it means the note crossed the peak
                    regression_model = LinearRegression()
                    # Fit the data(train the model)
                    y = chunk.data[:, np.newaxis]
                    x = np.arange(0, chunk.data.shape[0], 1)[:, np.newaxis]
                    regression_model.fit(x, y)
                    if regression_model.coef_ < 0:
                        find_max = Window(
                            dec, win_idx=(chunks[0].win_idx[0], chunks[-1].win_idx[1])
                        )
                        max_gloidx = find_max.win_idx[0] + find_max.max_idx()

                        #   #                   detect_start = max(max_gloidx - (25000+12500), 0)

                        detect_chunk = Window(
                            dec,
                            win_idx=(
                                max(max_gloidx - detect_win, idx),
                                min(max_gloidx + detect_win, dec.shape[0]),
                            ),
                        )
                        print(
                            detect_chunk.win_idx[0], detect_chunk.win_idx[1], max_gloidx
                        )
                        local_maxid = max_gloidx - detect_chunk.win_idx[0]
                        try:
                            start_idx = detect_chunk.first_cross(
                                cross_thres, increase=True, start_i=local_maxid
                            )
                            end_idx = detect_chunk.first_cross(
                                cross_thres, increase=False, start_i=local_maxid
                            )
                        except:
                            logger.info("First crossing failed...")
                            print("First crossing failed...")
                            logger.info(
                                "Will find the minimum value in the detection window..."
                            )
                            try:
                                start_idx, end_idx = detect_chunk.find_mins(
                                    local_maxid, thres
                                )
                                print(
                                    "Found",
                                    start_idx,
                                    end_idx,
                                    f"{detect_chunk.win_idx[0]/self.dec.shape[0]*100}%",
                                )
                            except Exception as e:
                                logger.error(e)
                                logger.error("Minimum search failed...stoping...")
                                print(f"N of detected notes: {count}")
                                self.notes = np.array(
                                    [start_idxs, end_idxs], dtype=np.int64
                                ).swapaxes(0, 1)
                                if return_result:
                                    return self.notes
                                return self
                        start_idxs.append(start_idx)
                        end_idxs.append(end_idx)
                        print(start_idx, end_idx)
                        count += 1
                        idx = end_idx
                        print(idx)
                        del chunks
                        break
                    chunk = Window(
                        dec,
                        win_idx=(chunks[-1].win_idx[-1], chunks[-1].win_idx[-1] + n_dp),
                    )
                if any(end_idxs):
                    idx = end_idxs[-1]
                else:
                    raise ValueError("this should never be raised")

            if idx >= dec.shape[0] - n_dp:
                logger.info(f"N of detected notes: {count}")
                print(f"N of detected notes: {count}")
                self.notes = np.array([start_idxs, end_idxs], dtype=np.int64).swapaxes(
                    0, 1
                )
                if return_result:
                    return self.notes
                return self


class Note:

    """Song Note"""

    def __init__(self, song: Song, si: int, ei: int):
        """TODO: to be defined."""
        self.song = song
        self.si = si
        self.ei = ei

    def find_maxamp(self):
        """ """
        # max after smoothing
        #       self.maxidx = self.si + np.argmax(np.abs(self.song.wf[self.si:self.ei]))
        # max before smoothing
        self.maxidx = self.si + np.argmax(self.song.dec[self.si : self.ei])

        return self

    def find_maxpower(self):
        """ """
        t = self.song.time[self.si : self.ei]
        self.tidxs = np.where((t[0] <= self.song.t) & (self.song.t <= t[-1]))[0]
        self.maxpw_idxs = [np.argmax(self.song.decpower[:, idx]) for idx in self.tidxs]
        return self

    def linearregression_aroundmaxamp(self):
        """ """
        regression_model = LinearRegression()
        self.x0 = np.array(self.song.t[self.tidxs])
        self.y0 = np.array(self.song.f[self.maxpw_idxs])
        # Fit the data(train the model)
        x = self.song.t[self.maxamp_ztidx - 3 : self.maxamp_ztidx + 1][:, np.newaxis]
        y = self.song.f[
            self.maxpw_idxs[
                self.maxamp_ztidx
                - 3
                - self.tidxs[0] : self.maxamp_ztidx
                + 1
                - self.tidxs[0]
            ]
        ][:, np.newaxis]
        regression_model.fit(x, y)
        self.regmodel = regression_model
        return self

    def find_distance_bw_maxpw_linearfit(self):
        self.dist = (
            np.abs(
                self.regmodel.coef_ * self.x0 - 1 * self.y0 + self.regmodel.intercept_
            )
            / np.sqrt(self.regmodel.coef_**2 + (-1) ** 2)
        ).squeeze()
        return self

    def filter_w_dist(self):
        """ """
        self.note_t = self.song.t[self.tidxs[self.dist < 0.004]]
        self.note_f = self.song.f[
            np.array(self.maxpw_idxs)[self.tidxs[self.dist < 0.004] - self.tidxs[0]]
        ]
        self.fmax = self.note_f[0]
        self.fmin = self.note_f[-1]
        self.fmod = self.fmax - self.fmin
        self.len = self.note_t[-1] - self.note_t[0]
        return self

    def detect_downsweep(self):
        """ """
        self.maxamp_ztidx = np.argmin(np.abs(self.song.time[self.maxidx] - self.song.t))
        return self

    def find_fmod(self):
        """ """
        self.find_maxamp()
        self.find_maxpower()
        self.detect_downsweep()
        self.linearregression_aroundmaxamp()
        self.find_distance_bw_maxpw_linearfit()
        self.filter_w_dist()
        return self

    def imshow(self, ax: Axes):
        """TODO: Docstring for imshow."""
        ax.imshow(
            self.song.power[:, self.tidxs],
            aspect="auto",
            origin="lower",
            extent=[
                self.song.t[self.tidxs[0]],
                self.song.t[self.tidxs[-1]],
                self.song.f[0],
                self.song.f[-1],
            ],
        )
        #       ax.plot(np.array(self.song.t[self.tidxs]),
        #               (np.array(self.song.t[self.tidxs])*self.regmodel.coef_ + self.regmodel.intercept_).squeeze(),
        #               color='white')
        ax.scatter(
            self.song.t[self.tidxs],
            self.song.f[self.maxpw_idxs],
            color="none",
            alpha=0.2,
            edgecolors="C0",
        )
        ax.scatter(
            self.song.t[self.maxamp_ztidx],
            40000,
            color="none",
            alpha=0.2,
            edgecolors="C1",
        )
        ax.scatter(
            self.song.t[self.tidxs[self.dist < 0.004]],
            self.song.f[
                np.array(self.maxpw_idxs)[self.tidxs[self.dist < 0.004] - self.tidxs[0]]
            ],
            color="none",
            alpha=0.2,
            edgecolors="C2",
        )
        return ax


def read_song(fpath: Path, singer: int = 1, load: bool = True) -> Song:
    """
    Read a file and return a Song object.
    """
    rate, wav = wavfile.read(fpath)
    return Song(wav, rate, singer=singer, fpath=fpath, load=load)


def read_audiosong(
    fpath: Path, rate: int = 10000, singer: int = 1, info: Info = None
) -> Song:
    """
    Read a file and return a Song object.
    """
    wf = np.load(fpath)
    audiosong = Song(wf, rate, singer=singer, fpath=fpath, info=info)
    try:
        audiosong.read_audionotes_file()
    except Exception as e:
        logger.warning(e)
    return audiosong


def audio_to_decibel(arr: ndarray, noise: ndarray) -> ndarray:
    """Convert 1D wave to decibel
    Returns:
    """
    if not arr.ndim == 1:
        raise ValueError("audio has to be a 1D array")
    return 10 * np.log10(abs(arr) / abs(noise).mean())


class Window(object):

    """Docstring for Window."""

    def __init__(self, arr: ndarray, win_idx: Tuple[int, int]):
        """"""
        self.data = arr[win_idx[0] : win_idx[1]]
        self.parent = arr
        self.win_idx = win_idx

    def max_bigger_than(self, val: float) -> bool:
        """Docstring for max_bigger_than.
        Returns: bool
        """
        return self.data.max() > val

    def max_smaller_than(self, val: float) -> bool:
        """Docstring for max_smaller_than.
        Returns: bool
        """
        return self.data.max() < val

    def min_bigger_than(self, val: float) -> bool:
        """Docstring for min_bigger_than.
        Returns: bool
        """
        return self.data.min() > val

    def min_smaller_than(self, val: float) -> bool:
        """Docstring for min_smaller_than.
        Returns: bool
        """
        return self.data.min() <= val

    def mean_smaller_than(self, val: float) -> bool:
        """
        Returns: bool
        """
        return self.data.mean() < val

    def min_idx(self) -> int:
        """Docstring for min_idx.
        Returns: int
        """
        min_where = np.where(self.data == self.data.min())[0]
        if len(min_where) != 1:
            print("Found multiple indices", min_where)
            print("Returning the smallest...")
            return int(min_where[0])
        return int(min_where)

    def max_idx(self) -> int:
        """Docstring for max_idx.
        Returns: int
        """
        max_where = np.where(self.data == self.data.max())[0]
        if len(max_where) != 1:
            print("Found multiple indices", max_where)
            print("Returning the smallest...")
            return int(max_where[0])
        return int(max_where)

    def parent_idx(self, idx: int) -> int:
        """Docstring for parent_idx.
        Returns: parent_idx (int)
        """
        return self.win_idx[0] + idx

    def first_cross(self, thres: float, increase: bool, start_i: int) -> int:
        """TODO: Docstring for first_cross.
        Returns: return idx in parent array
        of first crossing of thres
        """
        if increase:
            for pos in range(start_i):  # go backwards from start_i
                i = start_i - pos
                if self.data[i] <= thres:
                    idx = self.win_idx[0] + i
                    #                   print(f"Crossed at {idx}")
                    #                   logger.info("End crossed at %s", idx)
                    return idx
            raise ValueError("no crossing")
        for pos in range(self.data.shape[0])[start_i:]:
            # go foward from start_i
            i = pos
            if self.data[i] <= thres:
                idx = self.win_idx[0] + i
                #               print(f"Crossed at {idx}")
                #               logger.info("Start crossed at %s", idx)
                return idx
        raise ValueError("no crossing")

    def first_cross2(self, thres: float, increase: bool, start_i: int) -> int:
        """TODO: Docstring for first_cross.
        Returns: return idx in parent array
        of first crossing of thres
        """
        print("first cross2 start")
        ready = False
        if increase:
            for pos in range(start_i):  # go backwards from start_i
                i = start_i - pos
                if ready:
                    if self.data[i] >= thres:
                        idx = self.win_idx[0] + i
                        #                       print(f"Crossed at {idx}")
                        return idx
                if not ready and self.data[i] < 5:
                    ready_i = i
                    print("increase ready")
                    ready = True

            raise ValueError("no crossing")
        for pos in range(self.data.shape[0])[start_i:]:
            # go foward from start_i
            i = pos
            if ready:
                if self.data[i] >= thres:
                    idx = self.win_idx[0] + i
                    return idx
            if not ready and self.data[i] < 5:
                print("ready")
                ready = True

        plt.plot(self.data)
        plt.vlines([start_i, i], ymin=0, ymax=thres)
        plt.show()
        input()
        raise ValueError("no crossing")

    def find_mins(self, local_maxid: int, thres: float):
        """
        If first_cross fails, find the beginning and the end of a note
        by detecting local minimums
        """
        try:
            backward = self.first_cross2(thres, increase=True, start_i=local_maxid)
        except Exception as e:
            logger.error("backward cross %s", e)
            backward = 0
        try:
            start_local = np.argmin(self.data[backward:local_maxid])
        except Exception as e:
            logger.error("start_local %s", e)
            plt.plot(self.data)
            plt.vlines([backward, local_maxid], ymin=0, ymax=8, color="black")
            plt.show()
            input()
            start_local = np.argmin(self.data[:local_maxid])
        forward = self.first_cross2(thres, increase=False, start_i=local_maxid)
        try:
            end_local = np.argmin(self.data[local_maxid:forward])
        except Exception as e:
            logger.error("end_local %s", e)
            end_local = np.argmin(self.data[local_maxid:])
        #       start_local = np.argmin(self.data[:np.argmax(self.data)])
        #       end_local = np.argmin(self.data[np.argmax(self.data):])
        return start_local + self.win_idx[0], end_local + self.win_idx[0]


def find_song_from_datetime(
    arr: ndarray, dt: ndarray, target_dt: datetime, seconds: float, nidaq_freq: int
):
    """Docstring for find_song_from_datetime."""
    abs_diff = np.array([abs(diff.total_seconds()) for diff in dt - target_dt])
    idx = int(np.where(abs_diff == min(abs_diff))[0])
    beg_idx = idx - seconds * nidaq_freq  # 4000: sample rate of nidaq
    end_idx = idx + seconds * nidaq_freq
    return beg_idx, idx, end_idx


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


def detect_audio_notes(
    arr: ndarray,
    noise_win: Tuple[int, int] = (-125000, -1),
    n_smoothlen: int = 40,
    window: str = "hanning",
    thres: float = 5.0,
    cross_thres: float = 0.0,
):
    """Docstring for detect_audio_notes."""
    smo = smooth(abs(arr), window_len=n_smoothlen, window=window)
    try:
        dec = wave2decibel(smo, noise_win)
    except:
        import pdb

        pdb.set_trace()


def detect_notes(
    arr: ndarray,
    noise_win: Tuple[int, int] = None,
    noise_arr: ndarray = None,
    n_smoothlen=1000,
    window="hanning",
    thres: float = 5.0,
    cross_thres: float = 0.0,
    n_dp: int = 4000,
    detect_win: int = 25000 + 12500,
) -> ndarray:
    """
    Detect notes using linear regression.
    Returns
        note_idxs (ndarray): indices (within the song array) of the start
                             and end of all notes. Shape: (n_notes, 2)
    """
    logger.info(
        f"n_smoothlen: {n_smoothlen} thres: {thres}  cross_thres: {cross_thres}  n_dp: {n_dp}  detect_win: {detect_win}"
    )

    smo = smooth(abs(arr), window_len=n_smoothlen, window=window)
    try:
        dec = wave2decibel(smo, noise_win)
    except:
        import pdb

        pdb.set_trace()

    start_idxs: List[int] = []
    end_idxs: List[int] = []

    #   n_dp = 4000
    #   detect_win: int = 25000 + 12500

    idx = 0
    count = 0
    while True:
        if idx + n_dp <= arr.shape[0]:
            chunk = Window(dec, win_idx=(idx, idx + n_dp))
        else:
            chunk = Window(dec, win_idx=(idx, arr.shape[0]))
        if chunk.max_smaller_than(thres):
            # go next
            idx += n_dp
        else:
            chunks = []
            while True:
                chunks.append(chunk)
                # perform linear regression on the chunk
                # if coef > 0, the slope is still going up
                # when coef < 0, it means the note crossed the peak
                regression_model = LinearRegression()
                # Fit the data(train the model)
                y = chunk.data[:, np.newaxis]
                x = np.arange(0, chunk.data.shape[0], 1)[:, np.newaxis]
                regression_model.fit(x, y)
                if regression_model.coef_ < 0:
                    find_max = Window(
                        dec, win_idx=(chunks[0].win_idx[0], chunks[-1].win_idx[1])
                    )
                    max_gloidx = find_max.win_idx[0] + find_max.max_idx()

                    #                   detect_start = max(max_gloidx - (25000+12500), 0)

                    detect_chunk = Window(
                        dec,
                        win_idx=(
                            max(max_gloidx - detect_win, 0),
                            min(max_gloidx + detect_win, dec.shape[0]),
                        ),
                    )

                    local_maxid = max_gloidx - detect_chunk.win_idx[0]
                    try:
                        start_idx = detect_chunk.first_cross(
                            cross_thres, increase=True, start_i=local_maxid
                        )
                        end_idx = detect_chunk.first_cross(
                            cross_thres, increase=False, start_i=local_maxid
                        )
                    except:
                        logger.error("Issue with detecting notes. Ending...")
                        print("Issue with detecting notes. Ending...")
                        print(f"N of detected notes: {count}")
                        return np.array(
                            [start_idxs, end_idxs], dtype=np.int64
                        ).swapaxes(0, 1)
                    start_idxs.append(start_idx)
                    end_idxs.append(end_idx)
                    count += 1
                    idx = end_idx
                    del chunks
                    break
                chunk = Window(
                    dec, win_idx=(chunks[-1].win_idx[-1], chunks[-1].win_idx[-1] + n_dp)
                )
            if any(end_idxs):
                idx = end_idxs[-1]
            else:
                raise ValueError("this should never be raised")

        if idx >= dec.shape[0] - n_dp:
            logger.info(f"N of detected notes: {count}")
            print(f"N of detected notes: {count}")
            return np.array([start_idxs, end_idxs], dtype=np.int64).swapaxes(0, 1)


def _plot_song(
    song: ndarray,
    figpath: Path,
    suptitle: str,
    mouse_id: int,
    rate: int = 250000,
    note_idxs: ndarray = None,
    plot_dec: bool = False,
    dec: ndarray = None,
):
    """TODO: Docstring for plot_song.
    Returns: TODO

    """
    if not plot_dec and dec is None:
        raise ValueError("if plot_dec is True, then set dec.")
    fig, ax = plt.subplots(1, 1)
    t = np.linspace(0, song.shape[0] / rate, song.shape[0])
    if type(mouse_id) is not int:
        color = "black"
    else:
        color = f"C{mouse_id-1}"
    ax.plot(t, song, color=color)
    ax.set_ylabel("waveform", color=color)
    if plot_dec:
        ax2 = ax.twinx()
        ax2.plot(t, dec, color="gray", alpha=0.6)
        ax2.set_ylabel("dB", color="gray")
        ax2.set_ylim(0, 30)
    if note_idxs is not None:
        max_ = max(song)
        ax.vlines(x=t[note_idxs[:, 0]], ymin=0, ymax=max_, color="black")
        ax.vlines(x=t[note_idxs[:, 1]], ymin=0, ymax=max_, color="red")
    fig.suptitle(suptitle)
    fig.savefig(figpath)
    logger.info(f"Saved: {figpath}.")
    plt.close(fig)


def read_note_idxs_avi(arr_dir: Path, mouse_id: int):
    """Docstring for read_note_idxs.
    Returns:
        all_notes (List[ndarray()]): List of note_idxs (n_notes, 2)
                                     List size = n_songs
    """
    fpaths = natsorted(glob(str(arr_dir / f"M{mouse_id}" / f"*M{mouse_id}_notes.npy")))
    note_idxs: List[ndarray] = [np.load(fpath) for fpath in fpaths]
    dt0s: List[Timestamp] = [
        avifname_to_datetime(fpath.split("\\")[-1], 0) for fpath in fpaths
    ]
    return note_idxs, dt0s


def read_note_idxs(arr_dir: Path, mouse_id: int, audio: bool = False):
    """Docstring for read_note_idxs.
    Returns:
        all_notes (List[ndarray()]): List of note_idxs (n_notes, 2)
                                     List size = n_songs
    """
    if audio:
        fpaths = natsorted(glob(str(arr_dir / f"M{mouse_id}-*.npy")))
        song_ids = [
            int(fpath.split("\\")[-1].split(".")[0].split("-")[-1]) for fpath in fpaths
        ]
        #       n_files = len(glob(str(arr_dir / f"M{mouse_id}_audio_notes-*.npy")))
        return [np.load(arr_dir / f"M{mouse_id}_audio_notes-{i}.npy") for i in song_ids]
    #       return [np.load(arr_dir / f"M{mouse_id}_audio_notes-{i}.npy") for i in range(n_files)]
    else:
        n_files = len(glob(str(arr_dir / f"M{mouse_id}_notes-*.npy")))
        return [np.load(arr_dir / f"M{mouse_id}_notes-{i}.npy") for i in range(n_files)]


def note_lens(start_idxs: List[int], end_idxs: List[int], rate):
    """
    Calculate note lengths in seconds
    Returns:
    """
    first_idx = start_idxs[0]
    idxs = np.array((start_idxs, end_idxs))
    idxs -= first_idx
    seconds = np.zeros_like(idxs[0], dtype=np.float64)
    for i in range(idxs.shape[-1]):
        n_dp = idxs[1, i] - idxs[0, i] + 1
        seconds[i] = n_dp / rate
    return idxs, seconds


def find_songs_audio(arr: ndarray, dec: ndarray, rate: int, decimate: int = 1):
    """Docstring for find_songs_audio.
    arr (ndarray): (n_datapoint,)
    dec (ndarray): (n_datapoint,)
    decimate(int): decimate data by taking every {decimate} numbers.
                   default set to 1 (do not decimate)
    """
    if arr.shape != dec.shape:
        raise ValueError("Shapes of arr and dec have to match.")
    # decimate if necessary
    if decimate != 1:
        logger.info(f"decimating... taking every {decimate} points...")
        rate //= decimate
        arr = arr[::decimate]
        dec = dec[::decimate]
    chunk_dp = int(rate * 0.5)
    max_thres = 5
    #   max_thres = 7
    #   max_thres = 5
    #   thres = 5
    thres = 5
    #   thres = 5
    i = 0
    count = 0
    song = False
    songs = []
    song_decs = []
    start_idxs = []
    while i < arr.shape[0]:
        chunk = dec[i : i + chunk_dp]
        if (chunk > thres).any():
            logger.info(
                "Above threshold {str(pd.Timedelta(seconds=i/rate).to_pytimedelta())}"
            )
            if not song:
                start_i = i - chunk_dp * 2
                song = True
                quiet_count = 0
        elif song:
            if quiet_count < 2:
                quiet_count += 1
            elif quiet_count >= 2:
                end_i = i + chunk_dp
                # If the maximum does not exceed the max_thres,
                # it is probably not a song
                if (dec[start_i:end_i] > max_thres).any():
                    if dec[start_i:end_i].shape[0] > rate * 3:
                        s: ndarray = arr[start_i - rate * 4 : end_i + rate]
                        d: ndarray = dec[start_i - rate * 4 : end_i + rate]
                        songs.append(s)
                        song_decs.append(d)
                        start_idxs.append(start_i - rate * 4)
                        logger.info(
                            f"Above max threshold ({max_thres}) and longer than {rate*5} s. appended as a song {str(pd.Timedelta(seconds=start_i/rate).to_pytimedelta())}"
                        )
                        count += 1
                else:
                    logger.info(
                        f"Below max threshold ({max_thres}). Will be discarded..."
                    )
                song = False
                quiet_count = 0
        i += chunk_dp
    return songs, song_decs, start_idxs


def save_songs(
    songs: List[ndarray],
    start_idxs: List[int],
    mouse_id: int,
    save_dir: Path,
    song_idxs: List[str],
):
    """Docstring for save_songs.
    Returns: TODO
    """
    if len(songs) != len(start_idxs):
        raise ValueError("# of items in songs and start_idxs have to match.")
    if len(songs) != len(song_idxs):
        raise ValueError("# of items in songs and song_idxs have to match.")
    for i, song in enumerate(songs):
        arr_path = save_dir / f"M{mouse_id}-{song_idxs[i]}.npy"
        np.save(arr_path, songs[i])
        print(datetime.now(), "Saved:", arr_path)
        start_path = save_dir / f"M{mouse_id}_start-{song_idxs[i]}.npy"
        np.save(start_path, np.array([start_idxs[i]]))
        print(datetime.now(), "Saved:", start_path)


def find_songs_avisoft(wav: ndarray, rate: int, smooth_dp: int = 1000):
    """Find songs from a wav file.
    First, decimate and smooth the signal(40ms), then turn it into decibel.
    Seconds move a window (default 500ms) and detect ones containing
    values above a threshold. When threshold is not reached, consider it
    the end of the song.

    Args:
        arg1 (TODO): TODO

    Returns:
        songs (List[ndarray]): List of detected songs. (in original rate)
        start_idxs (List[int]): List of the first index of each song array.

    """
    # Decimate signal
    r = int(rate * 0.1)
    w = wav[::10]
    # Smooth (40ms)
    smo = smooth(abs(w), 1000, "hanning")
    dec = wave2decibel(smo, noise_win=(-r, -1))
    chunk_dp = int(r * 0.5)
    max_thres = 10
    thres = 2
    i = 0
    count = 0
    song = False
    songs = []
    start_idxs = []
    while i < w.shape[0]:
        chunk = dec[i : i + chunk_dp]
        if (chunk > thres).any():
            print("Above threshold", str(pd.Timedelta(seconds=i / r).to_pytimedelta()))
            if not song:
                start_i = i - chunk_dp * 2
                song = True
                quiet_count = 0
        elif song:
            if quiet_count < 2:
                quiet_count += 1
            elif quiet_count >= 2:
                end_i = i + chunk_dp
                # If the maximum does not exceed the max_thres,
                # it is probably not a song
                if (dec[start_i:end_i] > max_thres).any():
                    s: ndarray = wav[start_i * 10 - rate : end_i * 10 + rate]
                    print(s.shape)
                    songs.append(s)
                    start_idxs.append(start_i * 10 - rate + rate)
                    print(
                        f"Above max threshold ({max_thres})",
                        "appended as a song",
                        str(pd.Timedelta(seconds=start_i / r).to_pytimedelta()),
                    )
                    count += 1
                else:
                    print(f"Below max threshold ({max_thres})", "Will be discarded...")
                song = False
                quiet_count = 0
        i += chunk_dp
    return songs, start_idxs


def read_audio_section(filename, start_time, stop_time, dtype: str = None):
    """
    Read part of an audio file
    """
    track = sf.SoundFile(str(filename))

    can_seek = track.seekable()  # True
    if not can_seek:
        raise ValueError("Not compatible with seeking")

    sr = track.samplerate
    start_frame = int(np.rint(sr * start_time))
    frames_to_read = int(np.rint(sr * (stop_time - start_time)))
    track.seek(start_frame)
    if dtype is not None:
        audio_section = track.read(frames_to_read, dtype=dtype)
    else:
        audio_section = track.read(frames_to_read)
    return audio_section, sr


def read_audio_section_fromidx(
    filename: Path, start_idx: int, stop_idx: int, dtype: str = None
):
    """
    Read part of an audio file with indices
    """
    track = sf.SoundFile(str(filename))

    can_seek = track.seekable()  # True
    if not can_seek:
        raise ValueError("Not compatible with seeking")

    sr: int = track.samplerate
    frames_to_read: int = stop_idx - start_idx
    track.seek(start_idx)
    if dtype is not None:
        audio_section = track.read(frames_to_read, dtype=dtype)
    else:
        audio_section = track.read(frames_to_read)
    return audio_section, sr


def read_audio_songs_onoff(dir_path: Path, mouseid: int):
    """
    Read _audio_notes .npy files. Get start and end of songs.
    """
    fpaths = natsorted(glob(str(dir_path / f"M{mouseid}_audio_notes*.npy")))
    arr = np.zeros(shape=(len(fpaths), 2))
    for pos, fpath in enumerate(fpaths):
        a = np.load(fpath)
        arr[pos] = np.array([a[0, 0], a[-1, 1]])
    return arr


def write_audiosongs_json(song_dir: Path, take_name: str, mouseid: int = 1):
    """TODO: Docstring for write_nidaqstartdt.
    Returns: TODO

    """
    start_dt, end_dt, _, _ = read_mouseid_time(take_name)
    nidaq_dt0 = Timestamp(read_nidaq_dt0(take_name))
    fpaths: List[str] = natsorted(glob(str(song_dir / "*audio_notes-*.npy")))
    song_ids = [fpath.split(".")[-2].split("-")[-1] for fpath in fpaths]
    #   song_onoff = np.zeros(shape=(len(fpaths), 2))
    rate: int = 10000
    arr = read_audio_songs_onoff(song_dir, mouseid)
    song_onoffdt = np.zeros(arr.shape, dtype=Timestamp)
    song_onoffdelta = np.zeros(arr.shape, dtype=Timestamp)
    for pos, a in enumerate(arr):
        song_onoffdelta[pos, 0] = Timedelta(seconds=arr[pos, 0] / rate)
        song_onoffdelta[pos, 1] = Timedelta(seconds=arr[pos, 1] / rate)
        song_onoffdt[pos, 0] = song_onoffdelta[pos, 0] + nidaq_dt0
        song_onoffdt[pos, 1] = song_onoffdelta[pos, 1] + nidaq_dt0

    with open("audiosong.json", "r", encoding="utf-8") as file:
        data = json.load(file)
        take_json = data["info"][take_name]
        for pos, song_id in enumerate(song_ids):
            song_dict = {}
            song_dict["song_id"] = song_id
            song_dict["mouseid"] = str(mouseid)
            song_dict["start"] = str(song_onoffdt[pos, 0])
            song_dict["end"] = str(song_onoffdt[pos, 1])
            song_dict["start_d"] = str(song_onoffdelta[pos, 0])
            song_dict["end_d"] = str(song_onoffdelta[pos, 1])
            take_json[pos] = song_dict
            with open("audiosong.json", "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
                logger.info(f"added song to audiosong.json: {song_dict}")


def edit_mouseid_audiosongs(take_name: str, song_id: str, mouseid: str):
    """ """
    _, _, mouse1, mouse2 = read_mouseid_time(take_name)
    mouseids = [str(mouse1), str(mouse2)]
    if not str(mouseid) in mouseids:
        logger.warning(f"{mouseid} is not in {mouseids}")
    with open("audiosong.json", "r+") as f:
        data = json.load(f)
        info = data["info"]
        take_json = info[take_name]
        songid_dict: Dict[str, str] = {}
        for pos in take_json.keys():
            songid_dict[take_json[pos]["song_id"]] = pos
        logger.info(f"From {take_json[songid_dict[str(song_id)]]}")
        take_json[songid_dict[str(song_id)]]["mouseid"] = str(mouseid)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
        logger.info(f"To {take_json[songid_dict[str(song_id)]]}")


def edit_time_audiosongs(
    take_name: str,
    song_id: str,
    note_idxs: ndarray,
    nidaq_dt0: Timestamp,
    rate: int = 10000,
):
    """ """
    with open("audiosong.json", "r+") as f:
        data = json.load(f)
        info = data["info"]
        take_json = info[take_name]
        songid_dict = {}
        for pos in take_json.keys():
            songid_dict[take_json[pos]["song_id"]] = pos
        logger.info(f"From {take_json[songid_dict[str(song_id)]]}")
        start_d = Timedelta(seconds=note_idxs[0, 0] / rate)
        end_d = Timedelta(seconds=note_idxs[-1, 1] / rate)
        start = start_d + nidaq_dt0
        end = end_d + nidaq_dt0
        take_json[songid_dict[str(song_id)]]["start_d"] = str(start_d)
        take_json[songid_dict[str(song_id)]]["end_d"] = str(end_d)
        take_json[songid_dict[str(song_id)]]["start"] = str(start)
        take_json[songid_dict[str(song_id)]]["end"] = str(end)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
        logger.info(f"To {take_json[songid_dict[str(song_id)]]}")


def read_audiosongs_json(take_name):
    """TODO: Docstring for read_audiosongs_json.
    Returns: TODO

    """
    with open("audiosong.json", "r", encoding="utf-8") as file:
        data = json.load(file)
        take_json = data["info"][take_name]
    tobe_removed = []
    for key in take_json.keys():
        if Timedelta(take_json[key]["start_d"]) > Timedelta(hours=2):
            tobe_removed.append(key)
    for key in tobe_removed:
        take_json.pop(key)
    return take_json


def index_to_timestamp(index: int, daq_dt0: Timestamp, rate=10000):
    """Convert an index into a Timestamp"""
    return daq_dt0 + Timedelta(seconds=index / rate)


def timestamp_to_index(timestamp: Timestamp, daq_dt0: Timestamp, rate=10000):
    """Convert a Timestamp into an index"""
    return int((timestamp - daq_dt0).total_seconds() * rate)


def grab_audiochunk2(
    take_name: str,
    ts: float,
    te: float,
    file_dir: Path,
    rate: int = 10000,
    subj_idx: int = None,
):
    """
    Grab a chunk from the entired recording files. Return the chunk
    and the index of the start in the whole recording.
    """
    nidaq_dt0 = read_nidaq_dt0(take_name)
    rec_dt0, _, _, _ = read_mouseid_time(take_name, show=False)
    gap = Timestamp(rec_dt0) - Timestamp(nidaq_dt0)
    start_i = int((ts + gap.total_seconds()) * rate)
    end_i = int((te + gap.total_seconds()) * rate)
    unique_idxs = np.unique((start_i // (rate * 25), end_i // (rate * 25)))
    fids = np.arange(unique_idxs[0], unique_idxs[-1] + 1, 1).astype(int)
    fpaths = [file_dir / f"audio-{i}.npy" for i in fids]
    logger.info("Loading %s", fpaths)
    #   if len(fpaths) == 1:
    #       print('a')
    #       arr = np.load(fpaths[0])
    #   else:
    #       print('b')
    if subj_idx is None:
        arr = np.concatenate([np.load(fpath) for fpath in fpaths], axis=0)
    else:
        arr = np.concatenate([np.load(fpath)[subj_idx] for fpath in fpaths], axis=0)

    file_start = fids[0] * 25 * rate
    local_start = start_i - file_start
    local_end = end_i - file_start
    chunk = arr[local_start:local_end]
    return chunk, start_i


def grab_audiochunk(
    start_dt: Timestamp,
    end_dt: Timestamp,
    file_dir: Path,
    dt0: Timestamp,
    rate: int = 10000,
    subj_idx: int = None,
):
    """
    Grab a chunk from the entired recording files. Return the chunk
    and the index of the start in the whole recording.
    """
    start_i = timestamp_to_index(start_dt, dt0, rate)
    end_i = timestamp_to_index(end_dt, dt0, rate)
    unique_idxs = np.unique((start_i // (rate * 25), end_i // (rate * 25)))
    fids = np.arange(unique_idxs[0], unique_idxs[-1] + 1, 1)
    fpaths = [file_dir / f"audio-{i}.npy" for i in fids]
    logger.info(f"Loading {fpaths}")
    #   if len(fpaths) == 1:
    #       print('a')
    #       arr = np.load(fpaths[0])
    #   else:
    #       print('b')
    if subj_idx is None:
        arr = np.concatenate([np.load(fpath) for fpath in fpaths], axis=0)
    else:
        arr = np.concatenate([np.load(fpath)[subj_idx] for fpath in fpaths], axis=0)

    file_start = fids[0] * 25 * rate
    local_start = start_i - file_start
    local_end = end_i - file_start
    print(0)
    print(local_start, local_end)
    chunk = arr[local_start:local_end]
    print(chunk.shape)
    return chunk, start_i


def songids_of(mouseid: int, df: DataFrame):
    """ """
    song_idxs = np.unique(df["songid"].dropna().values).astype(int)
    song_ids: List[int] = []
    for song_id in song_idxs:
        singer = np.unique(df[df["songid"] == song_id]["singer"]).astype(int)
        if mouseid == singer:
            song_ids.append(song_id)
    return song_ids


def check_time(
    ts: Timestamp, time_from: Timestamp = None, time_to: Timestamp = None
) -> bool:
    """TODO: Docstring for _check_time.
    Returns: TODO
    """
    check: bool = True
    if time_from is None:
        pass
    elif time_from > ts:
        check = False
    if time_to is None:
        pass
    elif time_to < ts:
        check = False
    return check


def read_song_df(df, i: int, info: Info, load: bool = True, stft: bool = True):
    """TODO: Docstring for read_multi_song.
    Returns: TODO

    """
    mouse_id1 = int(df.loc[i, "singer"])
    #   mouse_id1 = int(df.loc[i, 'audio'].split("-")[0].split("M")[-1])
    song_id = df.loc[i, "audio"].split("-")[-1].split(".")[0]
    arr_dir = info.takedir / "song"
    logger.info("Read: %s", arr_dir / f"M{mouse_id1}-{song_id}_avi.wav")
    avisong = read_song(
        arr_dir / f"M{mouse_id1}-{song_id}_avi.wav", singer=mouse_id1, load=load
    )
    avisong._notesfpath = arr_dir / f"M{mouse_id1}_avi_notes-{song_id}.npy"
    avisong.maxidx_fpath = arr_dir / f"M{mouse_id1}_avi_maxidx-{song_id}.npy"
    avisong.imgfpath = str(avisong.fpath).split(".")[0]
    avisong.read_reference_file(df.loc[i, "reffpath"])
    if stft:
        avisong.stft()
    return avisong
