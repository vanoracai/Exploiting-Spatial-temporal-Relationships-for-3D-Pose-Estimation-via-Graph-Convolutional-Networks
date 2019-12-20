from tqdm import tqdm
from utils.utils1 import *
import data.common.eval_cal as eval_cal
import torch




def step(split, opt, actions, dataLoader, model, criterion, optimizer=None):#, key_index=None

    # initialize some definitions
    num_data_all = 0

    loss_all_sum = {'loss_gt':AccumLoss(), 'loss_diff':AccumLoss(), 'loss_sum':AccumLoss()}

    mean_error = {'xyz': 0.0, 'post': 0.0}
    error_sum = AccumLoss()


    if opt.dataset == 'h36m':
        limb_center = [2, 5, 11, 14] if opt.keypoints.startswith('sh') else [2, 5, 12, 15]
        limb_terminal = [3, 6, 12, 15] if opt.keypoints.startswith('sh') else [3,6,13,16]
        action_error_sum = define_error_list(actions)

        action_error_sum_post_out = define_error_list(actions)
        joints_left =[4, 5, 6, 10, 11, 12]  if opt.keypoints.startswith('sh') else [4,5,6,11,12,13]
        joints_right = [1, 2, 3, 13, 14, 15] if opt.keypoints.startswith('sh') else [1, 2, 3, 14, 15, 16]


    criterion_mse = criterion['MSE']
    model_st_gcn = model['st_gcn']
    model_post_refine = model['post_refine']


    if split == 'train':
        model_st_gcn.train()
        if opt.out_all:
            out_all_frame = True
        else:
            out_all_frame = False

    else:
        model_st_gcn.eval()
        out_all_frame = False

    torch.cuda.synchronize()



    # load data
    for i, data in enumerate(tqdm(dataLoader, 0)):


        if split == 'train':
            if optimizer is None:
                print("error! No Optimizer")
            else:
                optimizer_all = optimizer

        # load and process data
        batch_cam, gt_3D, input_2D, action, subject, scale , bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe (split,[input_2D, gt_3D, batch_cam, scale, bb_box])

        N = input_2D.size(0)
        num_data_all += N

        out_target = gt_3D.clone().view(N, -1, opt.n_joints, opt.out_channels)
        out_target[:, :, 0] = 0
        gt_3D = gt_3D.view(N, -1, opt.n_joints, opt.out_channels).type(torch.cuda.FloatTensor)

        if out_target.size(1) > 1:
            out_target_single = out_target[:,opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        # start forward process
        if opt.test_augmentation and split =='test':
            input_2D, output_3D = input_augmentation(input_2D, model_st_gcn, joints_left, joints_right)
        else:
            input_2D = input_2D.view(N, -1, opt.n_joints,opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor) # N, C, T, J, M
            output_3D = model_st_gcn(input_2D, out_all_frame)

        output_3D = output_3D.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, opt.n_joints, opt.out_channels)
        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1),opt.n_joints, opt.out_channels)
        if output_3D.size(1) > 1:
            output_3D_single = output_3D[:, opt.pad].unsqueeze(1)
        else:
            output_3D_single = output_3D

        if split == 'train':
            pred_out = output_3D # N, T, V, 3
        elif split == 'test':
            pred_out = output_3D_single

        # from uvd get xyz
        input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints ,2)


        if opt.crop_uv:
            pred_uv = back_to_ori_uv(input_2D, bb_box)
        else:
            pred_uv = input_2D

        uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
        xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
        xyz[:, :, 0, :] = 0

        if opt.post_refine:
            post_out = model_post_refine(output_3D_single, xyz)
            loss_sym = eval_cal.sym_penalty(opt.dataset, opt.keypoints, post_out)
            loss_post_refine = eval_cal.mpjpe(post_out, out_target_single) + 0.01*loss_sym
        else:
            loss_post_refine = 0


        #calculate loss
        loss_gt = eval_cal.mpjpe(pred_out, out_target)

        if opt.sym_penalty:
            loss_gt += 0.01 * eval_cal.sym_penalty(opt.dataset, opt.keypoints, pred_out)


        if opt.pad == 0 or split == 'test':
            loss_diff = torch.zeros(1).cuda()
        else:
            weight_diff = 4 * torch.ones(output_3D[:, :-1, :].size()).cuda()
            weight_diff[:, :, limb_center] = 2.5
            weight_diff[:, :, limb_terminal] = 1
            diff = (output_3D[:,1:] - output_3D[:,:-1]) * weight_diff
            loss_diff = criterion_mse(diff, Variable(torch.zeros(diff.size()), requires_grad=False).cuda())

        loss = loss_gt + opt.co_diff * loss_diff + loss_post_refine


        loss_all_sum['loss_gt'].update(loss_gt.detach().cpu().numpy() * N, N)
        loss_all_sum['loss_diff'].update(loss_diff.detach().cpu().numpy() * N, N)
        loss_all_sum['loss_sum'].update(loss.detach().cpu().numpy() * N, N)



        # train backpropogation
        if split == 'train':
            optimizer_all.zero_grad()
            loss.backward()
            optimizer_all.step()
            pred_out[:,:,0,:] = 0
            joint_error = eval_cal.mpjpe(pred_out,out_target).item()
            error_sum.update(joint_error*N, N)



        elif split == 'test':
            pred_out[:, :, 0, :] = 0
            action_error_sum = eval_cal.test_calculation(pred_out, out_target, action, action_error_sum,
                                                         opt.dataset, show_protocol2=opt.show_protocol2)

            if opt.post_refine:
                post_out[:, :, 0, :] = 0
                action_error_sum_post_out = eval_cal.test_calculation(post_out, out_target, action, action_error_sum_post_out, opt.dataset,
                                                             show_protocol2=opt.show_protocol2)

    if split == 'train':
        mean_error['xyz'] = error_sum.avg
        print('loss gt each frame of 1 sample: %f mm' % (loss_all_sum['loss_gt'].avg))
        print('loss diff of 1 sample: %f' % (loss_all_sum['loss_diff'].avg))
        print('loss of 1 sample: %f' % (loss_all_sum['loss_sum'].avg))
        print('mean joint error: %f' % (mean_error['xyz']*1000))

    elif split == 'test':
        if not opt.post_refine:
            mean_error_all = print_error(opt.dataset, action_error_sum, opt.show_protocol2)
            mean_error['xyz'] = mean_error_all

        elif opt.post_refine:
            print('-----post out')
            mean_error_all = print_error(opt.dataset, action_error_sum_post_out,  opt.show_protocol2)
            mean_error['post'] = mean_error_all

    return mean_error


def train(opt, actions,train_loader,model, criterion, optimizer):
    return step('train',  opt,actions,train_loader, model, criterion, optimizer)


def val( opt, actions,val_loader, model, criterion):
    return step('test',  opt,actions,val_loader, model, criterion)

def input_augmentation(input_2D, model_st_gcn, joints_left, joints_right):
    """
    for calculating augmentation results
    :param input_2D:
    :param model_st_gcn:
    :param joints_left: joint index of left part
    :param joints_right: joint index of right part
    :return:
    """
    N, _, T, J, C = input_2D.shape
    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) #N, C, T, J , M
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) #N, C, T, J , M

    # flip and reverse to original xyz
    output_3D_flip = model_st_gcn(input_2D_flip, out_all_frame=False)
    output_3D_flip[:, 0] *= -1
    output_3D_flip[:, :, :, joints_left + joints_right] = output_3D_flip[:, :, :, joints_right + joints_left]
    output_3D_non_flip = model_st_gcn(input_2D_non_flip, out_all_frame=False)

    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip

    return input_2D, output_3D