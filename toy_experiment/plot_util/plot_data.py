import matplotlib.pyplot as plt
import numpy as np
from baselines.common.plot_util import symmetric_ema
from baselines.common import plot_util as pu
import seaborn as sns

sns.set(style="darkgrid")

def xy_fn(r):
    x = r.x
    #y = r.y
    # y = pu.smooth(r.y, radius=40)
    y = r.y
    return x,y


class BaseData:
    def __init__(self, x = None, y=None, dir=''):
        self.dirname = dir
        self.x = x
        self.y = y


def append_data(x, res, label='trpo', x_var=None):
    for i in range(x.shape[1]):
        x_var_ = np.arange(x.shape[0]) if x_var is None else x_var[:, i]
        res.append(BaseData(x_var_, x[:, i], label))
    return res


def draw_one_res(x, label='trpo'):
    res = []
    for i in range(x.shape[1]):
        res.append(BaseData(np.arange(x.shape[0]), x[:, i], label))
    pu.plot_results(res, xy_fn=xy_fn, average_group=True)


def plot_res(res, **kwarg):
    xy_fn_ = xy_fn
    if 'xy_fn' in kwarg:
        xy_fn_ = kwarg['xy_fn']
        kwarg.pop('xy_fn')
    return pu.plot_results(res, xy_fn=xy_fn_, average_group=True, split_fn=lambda _: '', **kwarg)


def main():
    norig = 100
    res = []
    x = np.random.randn(1000, 6)
    #for i in range(6):
    #    xs = np.cumsum(np.random.rand(norig) * 10 / norig)
    #    yclean = np.sin(xs)
    #    ys = yclean + .6 * np.random.randn(yclean.size)
    #    res.append(BaseData(xs, ys, 'trpo'))
    # pu.plot_results(res, xy_fn=xy_fn, average_group=True)
    # pu.plt.show()
    res = []
    res = append_data(x, res, 'trpo')
    x = np.random.randn(1000, 6)
    res = append_data(x, res, 'ppo')
    f = plot_res(res, legend_outside=True)
    pu.plt.show()
    pu.plt.legend()
    print(f)
    f[0].savefig('foo.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
