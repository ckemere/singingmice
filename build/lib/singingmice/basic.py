#!/usr/bin/env python3
import errno
import os
import json
import logging

from datetime import datetime, timedelta
from pathlib import Path
import time
from typing.io import TextIO
from typing import List, Dict, Tuple, Iterable

from glob import glob
from tqdm import tqdm, trange
from natsort import natsorted

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from pandas import Timestamp, Timedelta
import pandas as pd

from scipy.io import wavfile


logger = logging.getLogger(__name__)



def mkdir(dir_path: Path, show: bool = False):
    """TODO: Docstring for mkdir.
    Returns: TODO
    """
    try:
        os.mkdir(dir_path)
        if show:
            print("Created:", dir_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    return dir_path


def avifname_to_datetime(fname: str, head_num: int = 1, f: bool = False):
    """Docstring for avifname_to_datetime.
    Returns:

    """
    date = fname[head_num : head_num + 10]
    hour = fname[head_num + 11 : head_num + 13]
    minute = fname[head_num + 14 : head_num + 16]
    if f:
        seconds = (
            fname[head_num + 17 : head_num + 19]
            + "."
            + fname[head_num + 20 : head_num + 26]
        )
    else:
        seconds = fname[head_num + 17 : head_num + 19]
    return pd.to_datetime(date + " " + hour + ":" + minute + ":" + seconds)


#   return pfname[18:20]d.to_datetime(fname[1:11] + " " + fname[12:14] + ":" + fname[15:17] + ":" + fname[18:20])


def integers2slices(integers: Iterable) -> List[slice]:
    """
    Group consecutive integers and make slice objects
    """
    if len(integers) == 0:
        logger.info("size 0. Returning an empty list.")
        return []
    slices = []
    beg: int = integers[0]
    keep = beg
    for idx in integers[1:]:
        if keep + 1 == idx:
            keep = idx
        else:
            sli = slice(beg, keep + 1)
            slices.append(sli)
            beg = idx
            keep = idx
    sli = slice(beg, keep + 1)
    slices.append(sli)
    logger.info("Found %s clusters.", len(slices))
    return slices
