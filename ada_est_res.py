import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AdaEstRes:
    scheme_name: Optional[str] = field(default=None)
    LS_flag: Optional[int] = field(default=None)
    h_init: float = field(default=np.nan)
    h: float = field(default=np.nan)
    h_opt: float = field(default=np.nan)
    r: float = field(default=np.nan)
    n_iter: Optional[int] = field(default=None)
    num_eval: Optional[int] = field(default=None)
    num_eval_no_cache: Optional[int] = field(default=None)
    estimated: float = field(default=np.nan)
    acc: float = field(default=np.nan)
    error: float = field(default=np.nan)
    rel_error: float = field(default=np.nan)
    eps_f: float = field(default=np.nan)
    r_l: float = field(default=np.nan)
    r_u: float = field(default=np.nan)

    def to_dict(self):
        return asdict(self)