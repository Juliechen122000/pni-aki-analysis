# -*- coding: utf-8 -*-
"""
轻量 Logistic 回归（IRLS + Wald 检验），用于替代 statsmodels
提供：系数、OR、95%CI、P 值、样本量、事件数
"""
import numpy as np
import pandas as pd
from scipy import stats


def logit_fit(X, y, ridge=1e-8, maxiter=60, tol=1e-9):
    """X: (n,p) 已含截距列; y: (n,) 0/1 -> (beta, cov, converged)"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)
    n, p = X.shape
    b = np.zeros(p)
    ok = True
    for _ in range(maxiter):
        eta = np.clip(X @ b, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        A = X.T @ (X * w[:, None]) + ridge * np.eye(p)
        rhs = X.T @ (w * z)
        try:
            bn = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            bn = np.linalg.lstsq(A, rhs, rcond=None)[0]
            ok = False
        if not np.all(np.isfinite(bn)):
            ok = False
            break
        if np.max(np.abs(bn - b)) < tol:
            b = bn
            break
        b = bn
    eta = np.clip(X @ b, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-9, None)
    A = X.T @ (X * w[:, None]) + ridge * np.eye(p)
    try:
        cov = np.linalg.pinv(A)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(np.nan_to_num(A) + 1e-6 * np.eye(p))
    if not np.all(np.isfinite(cov)):
        cov = np.diag(np.full(p, np.inf))
        ok = False
    return b, cov, ok


def design(df, formula_terms, ref=None):
    """
    极简设计矩阵构造
    formula_terms: list，元素可为
      ('num', col)      连续变量
      ('cat', col, refv) 分类变量（哑变量，refv 为参照）
    返回 (X, names)
    """
    cols = [np.ones(len(df))]
    names = ["Intercept"]
    for t in formula_terms:
        kind = t[0]
        if kind == "num":
            v = pd.to_numeric(df[t[1]], errors="coerce").astype(float)
            v = v.replace([np.inf, -np.inf], np.nan)
            v = v.fillna(v.median()).fillna(0.0)
            cols.append(v.values)
            names.append(t[1])
        else:
            col, refv = t[1], t[2]
            s = df[col].astype(str)
            lv = [x for x in sorted(s.dropna().unique()) if x != refv]
            for L in lv:
                cols.append((s == L).astype(float).values)
                names.append(f"{col}[{L}]")
    return np.column_stack(cols), names


def logistic_or(df, outcome, terms, exposure=None, dropna=None):
    """
    拟合 logistic，返回 dict:
      n, events, OR(暴露每降低1单位的OR), CI, P  以及各协变量结果
    exposure: 暴露变量名（连续）。若给定，额外返回"每降低1单位"的 OR
    dropna: 需要完整Case的列名列表
    """
    sub = df.copy()
    if dropna:
        sub = sub.dropna(subset=list(dropna))
    sub = sub[sub[outcome].notna()]
    if len(sub) < 50 or sub[outcome].nunique() < 2:
        return None
    X, names = design(sub, terms)
    y = sub[outcome].astype(float).values
    if y.sum() < 10 or y.sum() >= len(y):
        return None
    b, cov, ok = logit_fit(X, y)
    se = np.sqrt(np.diag(cov))
    z = b / np.where(se > 0, se, np.nan)
    pval = 2 * (1 - stats.norm.cdf(np.abs(z)))
    out = dict(n=int(len(sub)), events=int(y.sum()), converged=ok,
               coef=dict(zip(names, b)), se=dict(zip(names, se)),
               p=dict(zip(names, pval)))
    if exposure and exposure in names:
        i = names.index(exposure)
        beta = b[i]
        lo, hi = beta - 1.96 * se[i], beta + 1.96 * se[i]
        # PNI 每"降低" 1 单位 => OR = exp(-beta)
        out["OR_decrease"] = float(np.exp(-beta))
        out["CI_decrease"] = (float(np.exp(-hi)), float(np.exp(-lo)))
        out["P_exposure"] = float(pval[i])
        out["beta"] = float(beta)
    return out


def fmt_or(r):
    if r is None:
        return "NA"
    return "%.2f (%.2f-%.2f)" % (r["OR_decrease"], r["CI_decrease"][0], r["CI_decrease"][1])
