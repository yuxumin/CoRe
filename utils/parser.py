import os
import yaml
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type = str, choices=['MTL', 'Seven'], help = 'dataset')
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--fix_bn', type=bool, default=True)
    parser.add_argument('--resume', action='store_true', default=False ,help = 'autoresume training from exp dir(interrupted by accident)')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--Seven_cls', type = int, default=1, choices=[1,2,3,4,5,6], help = 'class idx in Seven')
    args = parser.parse_args()

    if args.test:
        if args.ckpts is None:
            raise RuntimeError('--ckpts should not be None when --test is activate')

    if args.benchmark == 'Seven':
        print(f'training CLASS idx {args.Seven_cls}')
        args.class_idx = args.Seven_cls
    return args

def setup(args):
    args.config = 'configs/{}_CoRe.yaml'.format(args.benchmark)
    args.experiment_path = os.path.join('./experiments', 'CoRe_RT', args.benchmark, args.exp_name)
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print("Failed to resume")
            args.resume = False
            setup(args)
            return

        print('Resume yaml from %s' % cfg_path)
        with open(cfg_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        merge_config(config, args)
        args.resume = True
    else:
        config = get_config(args)
        merge_config(config, args)
        create_experiment_dir(args)
        save_experiment_config(args)

def get_config(args):
    print('Load config yaml from %s' % args.config)
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def merge_config(config, args):
    for k, v in config.items():
        setattr(args, k, v)   

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    
def save_experiment_config(args):
    config_path = os.path.join(args.experiment_path,'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(args.__dict__, f)
        print('Save the Config file at %s' % config_path)