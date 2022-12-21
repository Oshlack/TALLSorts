#=======================================================================================================================
#
#   TALLSorts - Common functions
#   Author: Allen Gu, Breon Schmidt
#   License: MIT
#
#=======================================================================================================================

''' --------------------------------------------------------------------------------------------------------------------
Imports
---------------------------------------------------------------------------------------------------------------------'''

''' External '''
import os
from pathlib import Path

''' --------------------------------------------------------------------------------------------------------------------
Functions
---------------------------------------------------------------------------------------------------------------------'''


def get_project_root() -> Path:

    """
    Return the project root directory path relative to this file.

    ...

    Returns
    __________
    str
        Path to root directory
    """

    return Path(__file__).parent.parent

def message(message, level=False, important=False):

    """
    A simple way to print a message to the user with some formatting.

    ...

    Output
    __________
    A stylish, printed message.
    """

    text = "*** " + message + " ***" if important else message
    if level == 1:
        print("=======================================================================")
        print(text)
        print("=======================================================================")
    elif level == 2:
        print(text)
        print("-----------------------------------------------------------------------")
    elif level == "w":
        print("\n***********************************************************************")
        print(text)
        print("***********************************************************************\n")
    else:
        print(text)

def root_dir() -> Path:

    """
    Return the parent directory path relative to this file.

    ...

    Returns
    __________
    str
        Path to root directory
    """

    return Path(__file__).parent

def create_dir(path):

    """
    Create a directory given the supplied path.

    ...

    Output
    __________
    Directory created in the system
    """

    if isinstance(path, list):
        for p in path:
            try:
                os.mkdir(p)
            except OSError:
                continue
    else:
        try:
            os.mkdir(path)
        except OSError:
            pass