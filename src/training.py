import copy

import joblib
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from globals import cache, trackers
from optimizer import Optimizer
from util import w, detach_var, rsetattr, is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def update_optimizee(optimizee, result_params):
    with joblib.Parallel(n_jobs=2, prefer="threads") as parallel:
        parallel(joblib.delayed(rsetattr)(optimizee, name, result_params[name])
                 for name, p, in optimizee.all_named_parameters())


def optimizee_forward(optimizee, target):
    return optimizee(target)


def optimizer_forward(optimizer, gradients, hidden_states, cell_states):
    return optimizer(
        gradients,
        hidden_states,
        cell_states
    )


def calculate_gradients(loss):
    loss.backward(retain_graph=True)


def do_fit(opt_net,
           meta_opt,
           target_cls,
           target_to_opt,
           unroll,
           optim_it,
           out_mul,
           batch_size=128,
           should_train=True,
           tracker='default'):
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1

    target = target_cls(training=should_train, batch_size=batch_size)
    optimizee = w(target_to_opt())

    optimizee.register_forward_hook(trackers[tracker].forward)

    n_params = 0
    for name, p in optimizee.all_named_parameters():
        n_params += int(np.prod(p.size()))
    hidden_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    all_losses_ever = []
    if should_train:
        meta_opt.zero_grad()
    all_losses = None
    for iteration in range(1, optim_it + 1):
        trackers[tracker].start_timer('optimizee_predict')
        loss = optimizee_forward(optimizee, target)
        trackers[tracker].stop_timer('optimizee_predict')

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss

        all_losses_ever.append(loss.data.cpu().numpy())

        trackers[tracker].start_timer('calculate_gradients')
        calculate_gradients(loss)
        trackers[tracker].stop_timer('calculate_gradients')

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]

        trackers[tracker].start_timer('update_hidden_states')
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
            hs = [h[offset:offset + cur_sz] for h in hidden_states]
            cs = [c[offset:offset + cur_sz] for c in cell_states]
            trackers[tracker].start_timer('optimizer_predict')
            updates, new_hidden, new_cell = optimizer_forward(opt_net, gradients, hs, cs)
            trackers[tracker].stop_timer('optimizer_predict')
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset + cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset + cur_sz] = new_cell[i]
            result_params[name] = p + updates.view(*p.size()) * out_mul
            result_params[name].retain_grad()

            offset += cur_sz
        trackers[tracker].stop_timer('update_hidden_states')

        if iteration % unroll == 0:
            if should_train:
                trackers[tracker].start_timer('train_optimizer')
                meta_opt.zero_grad()
                all_losses.backward()
                meta_opt.step()
                trackers[tracker].stop_timer('train_optimizer')

            all_losses = None

            optimizee = w(target_to_opt())
            optimizee.load_state_dict(result_params)
            optimizee.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]

        else:
            trackers[tracker].start_timer('update_optimizee')
            update_optimizee(optimizee, result_params)
            trackers[tracker].stop_timer('update_optimizee')
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2

    return all_losses_ever


@cache.cache
def fit_optimizer(target_cls,
                  target_to_opt,
                  preproc=False,
                  unroll=20,
                  optim_it=100,
                  n_epochs=20,
                  n_tests=100,
                  lr=0.001,
                  batch_size=128,
                  iterations=20,
                  out_mul=1.0,
                  tracker='default'):
    opt_net = w(Optimizer(preproc=preproc))
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)

    best_net = None
    best_loss = 100000000000000000

    for _ in tqdm(range(n_epochs), 'epochs'):
        for _ in tqdm(range(iterations), 'iterations'):
            trackers[tracker].start_timer('fit_optimizer')
            do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, out_mul,
                   batch_size=batch_size,
                   should_train=True,
                   tracker=tracker)
            trackers[tracker].stop_timer('fit_optimizer')

        trackers[tracker].start_timer('calculate_loss')
        loss = (np.mean([
            np.sum(do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, out_mul,
                          batch_size=batch_size,
                          should_train=False,
                          tracker=tracker))
            for _ in tqdm(range(n_tests), 'tests')
        ]))
        trackers[tracker].stop_timer('calculate_loss')

        print(loss)
        if loss < best_loss:
            print(best_loss, loss)
            best_loss = loss
            best_net = copy.deepcopy(opt_net.state_dict())

    return best_loss, best_net


@cache.cache
def fit_normal(target_cls,
               target_to_opt,
               opt_class,
               n_tests=100,
               n_epochs=100,
               batch_size=128,
               tracker='default',
               **kwargs):
    results = []
    for i in tqdm(range(n_tests), 'tests'):
        target = target_cls(training=False, batch_size=batch_size)
        optimizee = w(target_to_opt())

        if tracker is not None:
            optimizee.register_forward_hook(trackers[tracker].forward)

        optimizer = opt_class(optimizee.parameters(), **kwargs)
        total_loss = []
        for _ in range(n_epochs):
            trackers[tracker].start_timer('optimizee_predict')
            loss = optimizee(target)
            trackers[tracker].stop_timer('optimizee_predict')

            total_loss.append(loss.data.cpu().numpy())

            trackers[tracker].start_timer('train_optimizer')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trackers[tracker].stop_timer('train_optimizer')
        results.append(total_loss)
    return results


@cache.cache
def get_fit_dict_test(n_tests,
                      opt_dict,
                      *args,
                      **kwargs):
    opt = w(Optimizer(preproc=True))
    opt.load_state_dict(opt_dict)
    np.random.seed(0)
    tracker = kwargs.get('tracker', 'default')
    trackers[tracker].start_timer('optimize')
    res = [do_fit(opt, *args, **kwargs) for _ in tqdm(range(n_tests), 'optimizer')]
    trackers[tracker].stop_timer('optimize')
    return res

