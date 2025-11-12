# -*- coding: utf-8 -*-
# rasotools/met/__init__.py
from . import convert
from . import errors
from . import humidity
from . import qc
from . import std
from . import time
from . import us_standard
from . import winds
from .__wrapper__ import *

# Lazy load allrasotrends only when accessed
import importlib

def __getattr__(name):
    if name == "allrasotrends":
        module = importlib.import_module(".allrasotrends", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")
