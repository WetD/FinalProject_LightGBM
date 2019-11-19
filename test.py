from hyperopt import fmin, tpe, hp, Trials

def ks_score(x):
    res = 11111
    for _ in range(10):
        res = x ** 2
    return res


trials = Trials()
best = fmin(ks_score, hp.uniform('x', -10, 10), algo=tpe.suggest, max_evals=10000, trials=trials)
