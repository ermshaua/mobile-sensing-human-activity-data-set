import numpy as np
import ruptures as rpt


def window(ts, window_size, n_cps, cost_func="mahalanobis", offset=0.05):
    transformer = rpt.Window(width=window_size, model=cost_func, min_size=int(ts.shape[0] * offset)).fit(ts)
    return np.array(transformer.predict(n_bkps=n_cps)[:-1], dtype=np.int64)

