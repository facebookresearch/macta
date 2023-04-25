# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from typing import Optional

import numpy as np


def load_trace(trace_file: str,
               limit: Optional[int] = None,
               legacy_trace_format: bool = False):
    with open(trace_file, "r") as f:
        lines = f.readlines()
        if limit is not None:
            lines = lines[:limit]

        data = []
        for line in lines:
            tokens = line.split()
            if legacy_trace_format:
                data.append(tokens)
            else:
                data.append((int(tokens[0]), int(tokens[3], 16)))

        if not legacy_trace_format:
            data = np.asarray(data, dtype=np.int64)

    return data
