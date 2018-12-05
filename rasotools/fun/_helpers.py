# -*- coding: utf-8 -*-

__all__ = ['find_files', 'now', 'message', 'dict2str', 'print_fixed']


def now():
    """ Datetime string

    Returns:
        str : datetime now
    """
    import datetime
    return datetime.datetime.now().isoformat()


def find_files(directory, pattern, recursive=True):
    """ find files

    Args:
        directory (str): directory path
        pattern (str):  regex string: '*.nc'
        recursive (bool): recursive search?

    Returns:
        list: of files
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


def dict2str(tmp):
    return ', '.join("{!s}={!r}".format(k, v) for (k, v) in tmp.items())


def print_fixed(liste, sep, width, offset=0):
    offset = " "*offset
    out = offset + liste[0]
    n = len(out)
    for i in liste[1:]:
        if (n + len(i) + 1) > width:
            out += sep + "\n" + offset + i
            n = len(offset + i)
        else:
            out += sep + " " + i
            n += len(i)+2

    return out
