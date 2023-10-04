"""
File: utils.py
Author: Yuki Fujishima
Email: yfujishima1001@gmail.com
Github: https://github.com/yufujis
Description: Library for basic operations used in this method.
"""
import errno
import os
import logging

from pathlib import Path
from typing import List


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
    return dir_path


def integers2slices(integers: List[int]) -> List[slice]:
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
