import torch
from torch.autograd import Variable
import numpy as np



import os

class AccumLoss(object):
    """
    for initialize and accumulate loss/err
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def get_varialbe(split,target):
    """

    :param split: 'train' or 'val'
    :param target: a list of tensors
    :return: a list of variables
    """
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        with torch.no_grad():
            for i in range(num):
                temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
                var.append(temp)

    return var




def get_uvd2xyz(uvd, gt_3D, cam):
    """
    transfer uvd to xyz

    :param uvd: N*T*V*3 (uv and z channel)
    :param gt_3D: N*T*V*3 (NOTE: V=0 is absolute depth value of root joint)

    :return: root-relative xyz results
    """
    N, T, V,_ = uvd.size()


    dec_out_all = uvd.view(-1, T, V, 3).clone()  # N*T*V*3
    root = gt_3D[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1).clone()# N*T*V*3
    enc_in_all = uvd[:, :, :, :2].view(-1, T, V, 2).clone()  # N*T*V*2

    cam_f_all = cam[..., :2].view(-1,1,1,2).repeat(1,T,V,1) # N*T*V*2
    cam_c_all = cam[..., 2:4].view(-1,1,1,2).repeat(1,T,V,1)# N*T*V*2

    # change to global
    z_global = dec_out_all[:, :, :, 2]# N*T*V
    z_global[:, :, 0] = root[:, :, 0, 2]
    z_global[:, :, 1:] = dec_out_all[:, :, 1:, 2] + root[:, :, 1:, 2]  # N*T*V
    z_global = z_global.unsqueeze(-1)  # N*T*V*1
    
    uv = enc_in_all - cam_c_all  # N*T*V*2
    xy = uv * z_global.repeat(1, 1, 1, 2) / cam_f_all  # N*T*V*2
    xyz_global = torch.cat((xy, z_global), -1)  # N*T*V*3
    xyz_offset = (xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1))# N*T*V*3


    return xyz_offset

def print_error(data_type,action_error_sum,show_protocol2=False):


    if data_type =='h36m':
        mean_error =  print_error_action(action_error_sum, show_protocol2)
    elif data_type.startswith('STB'):
        mean_error = print_error_directly(action_error_sum)

    return mean_error

def print_error_directly(action_error_sum):

    error = action_error_sum.avg * 1000.0
    print('Error:%f mm' % (error))
    return error



def print_error_action(action_error_sum, show_protocol2=False):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}
    if show_protocol2 :


        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))
        for action,value in action_error_sum.items():
            print("{0:<12} ".format(action), end="")
            for j in range(1,3):

                mean_error_each['p'+str(j)] = action_error_sum[action]['p'+str(j)].avg * 1000.0
                mean_error_all['p'+str(j)].update(mean_error_each['p'+str(j)], 1)

            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'],mean_error_each['p2']))
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg))
    else:

        print("{0:=^12} {1:=^6}".format("p#1 Action", "mm"))
        for action,value in action_error_sum.items():
            print("{0:<12} ".format(action), end="")

            mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
            print("{0:>6.2f}".format(mean_error_each['p1']))
            mean_error_all['p1'].update(mean_error_each['p1'], 1)


        print("{0:<12} {1:>6.2f}".format("Average", mean_error_all['p1'].avg))
    return mean_error_all['p1'].avg


def print_error_xyz(action_error_sum_xyz):
    mean_error_xyz_sum = np.zeros([3])
    print("{0:=^12} {1:=^6} {2:=^6} {3:=^6}".format("p#1 Action", "x", "y","z"))
    for action, value in action_error_sum_xyz.items():
        print("{0:<12} ".format(action), end="")
        mean_error_xyz = np.array(action_error_sum_xyz[action][1:4]) /action_error_sum_xyz[action][0] * 1000.0
        mean_error_xyz_sum += mean_error_xyz
        print("{0:>6.2f} {1:>6.2f} {2:>6.2f}".format(mean_error_xyz[0],mean_error_xyz[1],mean_error_xyz[2]))
    mean_error_xyz_sum/= float(len(action_error_sum_xyz))
    print("{0:<12} {1:>6.2f}{2:>6.2f}{3:>6.2f}".format("Average", mean_error_xyz_sum[0], mean_error_xyz_sum[1], mean_error_xyz_sum[2]))


def get_loss_sum(pre_list,value_list,num_data):
    """

    :param pre_list: [loss_sum1,loss_sum2,...]
    :param value_list: [loss_value_1,loss_value_2...]
    :param num_data: number of data in this batch
    :return:
    """
    num_list = len(pre_list)
    for i in range(num_list):
        pre_list[i] = pre_list[i] + value_list[i].detach() * num_data
    return pre_list


def save_model(previous_st_gcn_name, save_dir,epoch, save_out_type, data_threshold, model, model_name):
    if os.path.exists(previous_st_gcn_name):
        os.remove(previous_st_gcn_name)

    torch.save(model.state_dict(),
               '%s/model_%s_%d_eva_%s_%d.pth' % (save_dir, model_name, epoch, save_out_type, data_threshold * 100))
    previous_name = '%s/model_%s_%d_eva_%s_%d.pth' % (save_dir, model_name,epoch, save_out_type, data_threshold * 100)
    return previous_name


def define_error_list(actions):
    """
    define error sum_list
    error_sum: the return list
    actions: action list
    subjects: subjects list, if no subjects only make the list with actions
    :return: {action1:{'p1':, 'p2':},action2:{'p1':, 'p2':}}...
    """
    error_sum = {}
    error_sum.update({actions[i]: {'p1':AccumLoss(), 'p2':AccumLoss()} for i in range(len(actions))})
    return error_sum

def back_to_ori_uv(cropped_uv,bb_box):
    """
    for cropped uv, back to origial uv to help do the uvd->xyz operation
    :return:
    """
    N, T, V,_ = cropped_uv.size()
    uv = (cropped_uv+1)*(bb_box[:, 2:].view(N, 1, 1, 2)/2.0)+bb_box[:, 0:2].view(N, 1, 1, 2)
    return uv





























