import numpy as np
from enframe_06 import enframe


def STAc(x):
    """
    计算短时相关函数
    """
    para = np.zeros(x.shape)
    fn = x.shape[1]
    for i in range(fn):
        R = np.correlate(x[:, i], x[:, i], 'valid')
        para[:, i] = R
    return para


def STEn(x, win, inc):
    """
    计算短时能量函数
    """
    X = enframe(x, win, inc)
    s = np.multiply(X, X)
    return np.sum(s, axis=1)


def STMn(x, win, inc):
    """
    计算短时平均幅度计算函数
    """
    X = enframe(x, win, inc)
    s = np.abs(X)
    return np.mean(s, axis=1)


def STZcr(x, win, inc, delta=0):
    """
    计算短时过零率
    :param x:
    :param win:
    :param inc:
    :return:
    """
    absx = np.abs(x)
    x = np.where(absx < delta, 0, x)
    X = enframe(x, win, inc)
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    s = np.multiply(X1, X2)
    sgn = np.where(s < 0, 1, 0)
    return np.sum(sgn, axis=1)

