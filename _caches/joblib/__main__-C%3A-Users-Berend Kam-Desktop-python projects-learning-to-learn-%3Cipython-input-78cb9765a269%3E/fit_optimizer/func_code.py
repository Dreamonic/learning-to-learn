# first line: 1
@cache.cache
def fit_optimizer(target_cls, target_to_opt, preproc=False, unroll=20, optim_it=100, n_epochs=20, n_tests=100, lr=0.001, out_mul=1.0):
    opt_nets = []
    met_opts = []
    
    best_net = None
    best_loss = 100000000000000000
    
    for name, p in target_to_opt.all_named_parameters():
        i_size = int(np.prod(p.size()))
        opt_net = w(Optimizer(preproc=preproc), i_size = i_size)
        meta_opt = optim.Adam(opt_net.parameters(), lr=lr)
        opt_nets.append(opt_net)
        meta_opts.append(meta_opt)
    
    opts = zip(opt_nets, meta_opts)
    
    for _ in tqdm(range(n_epochs), 'epochs'):
        for _ in tqdm(range(20), 'iterations'):
            do_fit(opts, target_cls, target_to_opt, unroll, optim_it, n_epochs, out_mul, should_train=True)
        
        loss = (np.mean([
            np.sum(do_fit(opts, target_cls, target_to_opt, unroll, optim_it, n_epochs, out_mul, should_train=False))
            for _ in tqdm(range(n_tests), 'tests')
        ]))
        print(loss)
        if loss < best_loss:
            print(best_loss, loss)
            best_loss = loss
            best_net = copy.deepcopy(opt_net.state_dict())
            
    return best_loss, best_net
