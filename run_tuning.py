import time
from pathlib import Path

import numpy as np
from hyperopt import hp, fmin, tpe, rand
from pdfo import pdfo, Bounds
from sklearn.datasets import load_svmlight_file
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

LIBSVM = Path("libsvm").resolve(strict=True)


def load_libsvm(dataset):
    try:
        X_train, y_train = load_svmlight_file(LIBSVM / dataset)
        X_test, y_test = load_svmlight_file(LIBSVM / f"{dataset}.t")
    except FileNotFoundError:
        X_train, y_train = load_svmlight_file(LIBSVM / f"{dataset}.bz2")
        X_test, y_test = load_svmlight_file(LIBSVM / f"{dataset}.t.bz2")
    return X_train, y_train, X_test, y_test


def auc(X, y, **kwargs):
    svm = SVC()
    svm.set_params(**kwargs)
    cvs = cross_val_score(svm, X, y, scoring="roc_auc", n_jobs=-1)
    return np.mean(cvs)


def get_params(*args, exp=True):
    args = np.power(10, args) if exp else args
    return {"C": args[0], "gamma": args[1]}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    lb = np.array([-6.0, 0.0])
    ub = np.array([0.0, 3.0])
    x0 = np.log10(rng.uniform(10.0 ** lb, 10.0 ** ub))
    space = [hp.uniform("C", lb[0], ub[0]), hp.uniform("gamma", lb[1], ub[1])]
    x0_dict = get_params(*x0, exp=False)
    sl, su, sb = np.abs(x0 - lb), np.abs(x0 - ub), np.abs(ub - lb)
    rhobeg = 0.99 * np.min(np.r_[sl[sl > 0], su[su > 0], sb / 2.0, 1.0])
    max_eval = 100
    options = {"rhobeg": rhobeg}

    fline = "{:^8} {:^10} {:^13} {:^13} {:^13} {:^13}"
    rline = "{:^8} {:^10} {:^13.4e} {:^13.4e} {:^13.4e} {:^13.4e}"
    lsep = "-" * 75

    scaler = MaxAbsScaler(copy=False)
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean", copy=False)
    for dataset in ["splice", "svmguide1", "ijcnn1"]:
        X_train, y_train, X_test, y_test = load_libsvm(dataset)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        imputer.fit_transform(X_train)
        imputer.transform(X_test)
        print("Dataset: {}".format(dataset))
        print(lsep)
        print(fline.format("Solver", "No. eval", "AUC score", "Accuracy", "Exec. time", "Optim. time"))
        print(lsep)


        def loss(args, eval_time):
            t0 = time.time()
            auc_avg = auc(X_train, y_train, **get_params(*args))
            eval_time.append(time.time() - t0)

            return 1. - auc_avg

        # PDFO
        t0 = time.time()
        eval_time = []
        options["maxfev"] = max_eval
        res = pdfo(lambda x: loss(x, eval_time), x0, bounds=Bounds(lb, ub), options=options)
        elapsed = time.time() - t0
        svm = SVC(**get_params(*res.x), probability=True, random_state=0)
        svm.fit(X_train, y_train)
        auc_score = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])
        acc_score = accuracy_score(y_test, svm.predict(X_test))
        print(rline.format("PDFO", res.nfev, auc_score, acc_score, elapsed, elapsed - sum(eval_time)))

        # Random Search
        for k in [1, 2, 3]:
            rng = np.random.default_rng(0)
            t0 = time.time()
            eval_time = []
            options["maxfev"] = k * max_eval
            res = fmin(lambda x: loss(x, eval_time), space, rand.suggest, options["maxfev"] - 1, rstate=rng, points_to_evaluate=[x0_dict], verbose=False)
            elapsed = time.time() - t0
            svm = SVC(**get_params(*res.values()), probability=True, random_state=0)
            svm.fit(X_train, y_train)
            auc_score = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])
            acc_score = accuracy_score(y_test, svm.predict(X_test))
            print(rline.format("RS", options["maxfev"], auc_score, acc_score, elapsed, elapsed - sum(eval_time)))

        # TPE
        for k in [1, 3]:
            rng = np.random.default_rng(0)
            t0 = time.time()
            eval_time = []
            options["maxfev"] = k * max_eval
            res = fmin(lambda x: loss(x, eval_time), space, tpe.suggest, options["maxfev"] - 1, rstate=rng, points_to_evaluate=[x0_dict], verbose=False)
            elapsed = time.time() - t0
            svm = SVC(**get_params(*res.values()), probability=True, random_state=0)
            svm.fit(X_train, y_train)
            auc_score = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])
            acc_score = accuracy_score(y_test, svm.predict(X_test))
            print(rline.format("TPE", options["maxfev"], auc_score, acc_score, elapsed, elapsed - sum(eval_time)))

        print(lsep)
