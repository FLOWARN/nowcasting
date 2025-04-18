from torch import optim

optim_parameters = {
    'adam': ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad'],
}


def get_optim_scheduler(config, model, epochs):
    parameters = model.parameters()
    
    opt_lower = config['opt'].lower()
    opt_kwargs_temp = dict.fromkeys(optim_parameters[opt_lower])
    
    for k in opt_kwargs_temp.keys(): 
        b = f"opt_{k}" if f"opt_{k}" in config else k
        opt_kwargs_temp[k] = config[b] if (b in config) else None

    opt_kwargs = {k: v for k, v in opt_kwargs_temp.items() if v is not None}


    if opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_kwargs)
    else:
        assert False and "Invalid optimizer"

    sched_lower = config['sched'].lower()
    total_steps = epochs * config['steps_per_epoch']
    by_epoch = True
    if sched_lower == 'onecycle':
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            total_steps=total_steps,
        )
        by_epoch = False
    else:
        assert False and "Invalid scheduler"

    return optimizer, lr_scheduler, by_epoch