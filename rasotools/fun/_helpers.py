# -*- coding: utf-8 -*-

__all__ = ['find_files', 'now', 'message']


def now():
    """
    Functions returns the current date string
    Returns
    -------
    str
        current date
    """
    import datetime
    return datetime.datetime.now().isoformat()


def find_files(directory, pattern, recursive=True):
    """ Find files in a directory give a pattern

    Parameters
    ----------
    directory       str
    pattern         str
    recursive       bool

    Returns
    -------
    list
    """
    import os
    import fnmatch
    matches = []
    if recursive:
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(root, filename))
    else:
        matches.extend(fnmatch.filter([os.path.join(directory, ifile) for ifile in os.listdir(directory)], pattern))
    return matches


def message(*args, mname=None, verbose=0, level=0, logfile=None, **kwargs):
    """

    Parameters
    ----------
    args : list
    mname : str
        Message Name
    level : int
        Level of Messaging
    logfile : str
        filename of Logfile to write to
    verbose : int
        verboseness in combination with level
    kwargs : dict

    Returns
    -------

    """
    if logfile is not None:
        # with open(kwargs['filename'], 'a' if not kwargs.get('force', False) else 'w') as f:
        with open(logfile, 'a') as f:
            f.write(_print_string(*args, **kwargs) + "\n")

    elif verbose > level:
        text = _print_string(*args, **kwargs)
        if mname is not None:
            text = "[%s] " % mname + text

        print(text)
    else:
        pass


def _print_string(*args, adddate=False, **kwargs):
    if adddate:
        return "[" + now() + "] " + " ".join([str(i) for i in args])
    else:
        return " ".join([str(i) for i in args])

