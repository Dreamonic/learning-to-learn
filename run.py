import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.optim as optim

from globals import trackers
from models import MNISTLoss, MNISTNet
from training import fit_optimizer, get_fit_dict_test, fit_normal
from util import show_and_save_plot, markup_plot

sns.set(color_codes=True)
sns.set_style("white")

BATCH_SIZE = 128

NORMAL_OPTS = [(optim.Adam, {}), (optim.RMSprop, {}), (optim.SGD, {'momentum': 0.9}),
               (optim.SGD, {'nesterov': True, 'momentum': 0.9})]
OPT_NAMES = ['ADAM', 'RMSprop', 'SGD', 'NAG']

QUAD_LRS = [0.03, 0.01, 1.0, 1.0]

trackers['Training'].start_timer('total')

loss, mnist_optimizer = fit_optimizer(MNISTLoss, MNISTNet, lr=0.01, n_epochs=1, n_tests=1, out_mul=0.1,
                                      batch_size=BATCH_SIZE,
                                      iterations=1,
                                      preproc=True,
                                      tracker='Training')

N_TESTS = 1

fit_data = np.zeros((N_TESTS, 200, len(OPT_NAMES) + 1))

for i, ((opt, extra_kwargs), lr) in enumerate(zip(NORMAL_OPTS, QUAD_LRS)):
    np.random.seed(0)
    fit_data[:, :, i] = np.array(
        fit_normal(MNISTLoss, MNISTNet, opt, lr=lr, n_tests=N_TESTS, n_epochs=200, batch_size=BATCH_SIZE,
                   tracker=OPT_NAMES[i],
                   **extra_kwargs))

fit_data[:, :, len(OPT_NAMES)] = np.array(
    get_fit_dict_test(N_TESTS, mnist_optimizer, None, MNISTLoss, MNISTNet, 1, 200, out_mul=0.1,
                      batch_size=BATCH_SIZE,
                      should_train=False,
                      tracker='LSTM'))

trackers['Training'].stop_timer('total')

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(10, 5, forward=True)

ax1 = sns.tsplot(data=fit_data[:, :100, :], condition=OPT_NAMES + ['LSTM'], linestyle='--',
                 color=['#ea4c4e', '#377eb8', '#4eae4b', '#994fa1', '#ff8101'], ax=ax1)

ax2 = sns.tsplot(data=fit_data[:, 100:, :], condition=OPT_NAMES + ['LSTM'], linestyle='--',
                 color=['#ea4c4e', '#377eb8', '#4eae4b', '#994fa1', '#ff8101'], ax=ax2)

markup_plot(ax1, 'MNIST', domain=range(20, 101, 20))
markup_plot(ax2, 'MNIST, 200 steps', domain=range(20, 101, 20), leg=False)

show_and_save_plot('MNIST', path='mnist2', markup=False)

for t in trackers.values():
    t.save()
