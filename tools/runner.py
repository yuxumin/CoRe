import numpy as np
import torch
import torch.nn as nn

from scipy import stats
from tools import builder, helper
from utils import misc
import time

def test_net(args):
    print('Tester start ... ')
    train_dataset, test_dataset = builder.dataset_builder(args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                            shuffle=False,num_workers = int(args.workers),
                                            pin_memory=True)
    base_model, regressor = builder.model_builder(args)
    # load checkpoints
    builder.load_model(base_model, regressor, args)

    # if using RT, build a group
    group = builder.build_group(train_dataset, args)

    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        regressor = regressor.cuda()
        torch.backends.cudnn.benchmark = True

    #  DP
    base_model = nn.DataParallel(base_model)
    regressor = nn.DataParallel(regressor)

    test(base_model, regressor, test_dataloader, group, args)

def run_net(args):
    print('Trainer start ... ')
    # build dataset
    train_dataset, test_dataset = builder.dataset_builder(args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs_train,
                                            shuffle=True,num_workers = int(args.workers),
                                            pin_memory=True, worker_init_fn=misc.worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                            shuffle=False,num_workers = int(args.workers),
                                            pin_memory=True)
    # build model
    base_model, regressor = builder.model_builder(args)

    # if using RT, build a group
    group = builder.build_group(train_dataset, args)
    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_model = base_model.cuda()
        regressor = regressor.cuda()
        torch.backends.cudnn.benchmark = True

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, regressor, args)

    # parameter setting
    start_epoch = 0
    global epoch_best, rho_best, L2_min, RL2_min
    epoch_best = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    # resume ckpts
    if args.resume:
        start_epoch, epoch_best, rho_best, L2_min, RL2_min = \
            builder.resume_train(base_model, regressor, optimizer, args)
        print('resume ckpts @ %d epoch( rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (start_epoch - 1, rho_best,  L2_min, RL2_min))

    #  DP
    base_model = nn.DataParallel(base_model)
    regressor = nn.DataParallel(regressor)

    # loss
    mse = nn.MSELoss().cuda()
    nll = nn.NLLLoss().cuda()

    # trainval

    # training
    for epoch in range(start_epoch, args.max_epoch):
        true_scores = []
        pred_scores = []
        num_iter = 0
        base_model.train()  # set model to training mode
        regressor.train()
        if args.fix_bn:
            base_model.apply(misc.fix_bn)  # fix bn
        for idx, (data , target) in enumerate(train_dataloader):
            # break
            num_iter += 1
            opti_flag = False

            true_scores.extend(data['final_score'].numpy())
            # data preparing
            # video_1 is the test video ; video_2 is exemplar
            if args.benchmark == 'MTL':
                video_1 = data['video'].float().cuda() # N, C, T, H, W
                if args.usingDD:
                    label_1 = data['completeness'].float().reshape(-1,1).cuda()
                    label_2 = target['completeness'].float().reshape(-1,1).cuda()
                else:
                    label_1 = data['final_score'].float().reshape(-1,1).cuda()
                    label_2 = target['final_score'].float().reshape(-1,1).cuda()
                if not args.dive_number_choosing and args.usingDD:
                    assert (data['difficulty'].float() == target['difficulty'].float()).all()
                diff = data['difficulty'].float().reshape(-1,1).cuda()
                video_2 = target['video'].float().cuda() # N, C, T, H, W

            elif args.benchmark == 'Seven':
                video_1 = data['video'].float().cuda() # N, C, T, H, W
                label_1 = data['final_score'].float().reshape(-1,1).cuda()
                video_2 = target['video'].float().cuda()
                label_2 = target['final_score'].float().reshape(-1,1).cuda()
                diff = None
            else:
                raise NotImplementedError()
            # forward
            if num_iter == args.step_per_update:
                num_iter = 0
                opti_flag = True

            helper.network_forward_train(base_model, regressor, pred_scores, video_1, label_1, video_2, label_2, diff, group, mse, nll, optimizer, opti_flag, epoch, idx+1, len(train_dataloader), args)

        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores,2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()) ,2).sum() / true_scores.shape[0]
        print('[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f, lr1: %.4f, lr2: %.4f'%(epoch, rho, L2, RL2, optimizer.param_groups[0]['lr'],  optimizer.param_groups[1]['lr']))


        validate(base_model, regressor, test_dataloader, epoch, optimizer, group, args)
        helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, 'last', args)
        print('[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f'%(epoch, rho_best, L2_min, RL2_min))
        # scheduler lr
        if scheduler is not None:
            scheduler.step()

def validate(base_model, regressor, test_dataloader, epoch, optimizer, group, args):
    print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best, rho_best, L2_min, RL2_min
    true_scores = []
    pred_scores = []
    base_model.eval()  # set model to eval mode
    regressor.eval()
    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()
        for batch_idx,  (data , target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()
            true_scores.extend(data['final_score'].numpy())
            # data prepare
            if args.benchmark == 'MTL':
                video_1 = data['video'].float().cuda() # N, C, T, H, W
                if args.usingDD:
                    label_2_list = [item['completeness'].float().reshape(-1,1).cuda() for item in target]
                else:
                    label_2_list = [item['final_score'].float().reshape(-1,1).cuda() for item in target]
                diff = data['difficulty'].float().reshape(-1,1).cuda()
                video_2_list = [item['video'].float().cuda() for item in target]
                # check
                if not args.dive_number_choosing and args.usingDD:
                    for item in target:
                        assert (diff == item['difficulty'].float().reshape(-1,1).cuda()).all()
            elif args.benchmark == 'Seven':
                video_1 = data['video'].float().cuda() # N, C, T, H, W
                video_2_list = [item['video'].float().cuda() for item in target]
                label_2_list = [item['final_score'].float().reshape(-1,1).cuda() for item in target]
                diff = None
            else:
                raise NotImplementedError()
            helper.network_forward_test(base_model, regressor, pred_scores, video_1, video_2_list, label_2_list, diff, group, args)
            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d][%d/%d] \t Batch_time %.2f \t Data_time %.2f '
                    % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores,2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()) ,2).sum() / true_scores.shape[0]
        if L2_min > L2:
            L2_min = L2
        if RL2_min > RL2:
            RL2_min = RL2
        if rho > rho_best:
            rho_best = rho
            epoch_best = epoch
            print('-----New best found!-----')
            helper.save_outputs(pred_scores, true_scores, args)
            helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, 'best', args)

        print('[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f'%(epoch, rho, L2, RL2))

def test(base_model, regressor, test_dataloader, group, args):
    global use_gpu
    true_scores = []
    pred_scores = []
    base_model.eval()  # set model to eval mode
    regressor.eval()
    batch_num = len(test_dataloader)
    with torch.no_grad():
        datatime_start = time.time()
        for batch_idx,  (data , target) in enumerate(test_dataloader, 0):
            datatime = time.time() - datatime_start
            start = time.time()
            true_scores.extend(data['final_score'].numpy())
            # data prepare
            if args.benchmark == 'MTL':
                video_1 = data['video'].float().cuda() # N, C, T, H, W
                if args.usingDD:
                    label_2_list = [ item['completeness'].float().reshape(-1,1).cuda() for item in target]
                else:
                    label_2_list = [ item['final_score'].float().reshape(-1,1).cuda() for item in target]
                diff = data['difficulty'].float().reshape(-1,1).cuda()
                video_2_list = [item['video'].float().cuda() for item in target]
                # check
                if not args.dive_number_choosing and args.usingDD:
                    for item in target:
                        assert (diff == item['difficulty'].float().reshape(-1,1).cuda()).all()
            elif args.benchmark == 'Seven':
                video_1 = data['video'].float().cuda() # N, C, T, H, W
                video_2_list = [ item['video'].float().cuda() for item in target]
                label_2_list = [ item['final_score'].float().reshape(-1,1).cuda() for item in target]
                diff = None
            else:
                raise NotImplementedError()
            helper.network_forward_test(base_model, regressor, pred_scores, video_1, video_2_list, label_2_list, diff, group, args)
            batch_time = time.time() - start
            if batch_idx % args.print_freq == 0:
                print('[TEST][%d/%d] \t Batch_time %.2f \t Data_time %.2f '
                    % (batch_idx, batch_num, batch_time, datatime))
            datatime_start = time.time()
        # analysis on results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores,2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()) ,2).sum() / true_scores.shape[0]
        print('[TEST] correlation: %.6f, L2: %.6f, RL2: %.6f'%(rho, L2, RL2))
