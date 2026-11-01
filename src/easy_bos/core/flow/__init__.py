"""Displacement estimators. Importing this package registers every method in ``REGISTRY``."""

from easy_bos.core.flow import correlation, lucas_kanade, opencv_flow, raft, variational, wavelet
from easy_bos.core.flow.base import REGISTRY, FlowMethod, FlowResult, create

__all__ = ["REGISTRY", "FlowMethod", "FlowResult", "create", "correlation", "lucas_kanade",
           "opencv_flow", "raft", "variational", "wavelet"]
