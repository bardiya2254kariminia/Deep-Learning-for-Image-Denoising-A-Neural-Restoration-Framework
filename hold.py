from collections import defaultdict
import numpy as np
from argparse import Namespace

a = defaultdict(lambda: [])
b = defaultdict(lambda: [])
b["a"].extend([1,2,3])
b["b"].extend([1,2,3])
b["c"].extend([1,2,3])

a = Namespace(**{k : np.mean(v) for k,v in b.items()})
print(a.a)