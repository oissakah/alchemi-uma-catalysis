#!/usr/bin/env python
from __future__ import annotations

import pickle
import pprint
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    obj = pickle.load(f)

print("type:", type(obj))
try:
    print("length:", len(obj))
except Exception:
    pass

if isinstance(obj, dict):
    print("first keys:", list(obj)[:10])
    for k in list(obj)[:3]:
        print("\nKEY:", k)
        pprint.pp(obj[k])
else:
    pprint.pp(obj if not hasattr(obj, "__len__") else obj[:3])
