import math
import torch
import numpy as np


def reserve_schedule_sampling_exp(itr, batch_size, args):
    T, img_channel, img_height, img_width = args.in_shape
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (batch_size, args.pre_seq_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (batch_size, args.aft_seq_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((img_height // args.patch_size,
                    img_width // args.patch_size,
                    args.patch_size ** 2 * img_channel))
    zeros = np.zeros((img_height // args.patch_size,
                      img_width // args.patch_size,
                      args.patch_size ** 2 * img_channel))

    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - 2):
            if j < args.pre_seq_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.pre_seq_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - 2,
                                  img_height // args.patch_size,
                                  img_width // args.patch_size,
                                  args.patch_size ** 2 * img_channel))
    return torch.FloatTensor(real_input_flag).to(args.device)


def schedule_sampling(eta, itr, batch_size, config):
    img_channel, img_height, img_width = config['channels'], config['img_height'], config['img_width']
    patch_size = config['patch_size']
    zeros = np.zeros((batch_size,
                      config['out_seq_length'] - 1,
                      img_height // config['patch_size'],
                      img_width // config['patch_size'],
                      config['patch_size'] ** 2 * img_channel))
    if not config['scheduled_sampling']:
        return 0.0, zeros

    if itr < config['sampling_stop_iter']:
        eta -= config['sampling_changing_rate']
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (batch_size, config['out_seq_length'] - 1))
    true_token = (random_flip < eta)
    ones = np.ones((img_height // patch_size,
                    img_width // patch_size,
                    patch_size ** 2 * img_channel))
    zeros = np.zeros((img_height // patch_size,
                      img_width // patch_size,
                      patch_size ** 2 * img_channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(config['out_seq_length'] - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  config['out_seq_length'] - 1,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  patch_size ** 2 * img_channel))
    return eta, torch.FloatTensor(real_input_flag).to(config['device'])



def reshape_patch(img_tensor, patch_size, channel_sep=False):
    assert 5 == img_tensor.ndim
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape

    if (channel_sep) and (num_channels> 1) :
        img_tensor_list = []
        for i in range(num_channels):
            img_tensor_i = img_tensor[:, :, :, :, i:i+1]

            a = img_tensor_i.reshape(batch_size, seq_length,
                                        img_height//patch_size, patch_size,
                                        img_width//patch_size, patch_size,
                                        1)
            b = a.transpose(3, 4)
            img_tensor_i = b.reshape(batch_size, seq_length,
                                        img_height//patch_size,
                                        img_width//patch_size,
                                        patch_size*patch_size*1)
            
            img_tensor_list.append(img_tensor_i)
        
        patch_tensor = torch.cat(img_tensor_list, -1)

    else:
        a = img_tensor.reshape(batch_size, seq_length,
                                    img_height//patch_size, patch_size,
                                    img_width//patch_size, patch_size,
                                    num_channels)
        b = a.transpose(3, 4)
        patch_tensor = b.reshape(batch_size, seq_length,
                                    img_height//patch_size,
                                    img_width//patch_size,
                                    patch_size*patch_size*num_channels)
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size, channel_sep=False):
    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape
    img_channels = channels // (patch_size*patch_size)

    if (channel_sep) and (img_channels > 1):
        img_tensor_list = []
        for i in range(img_channels):
            patch_tensor_i = patch_tensor[:, :, :, :, i*patch_size*patch_size:(i+1)*patch_size*patch_size]
            a = patch_tensor_i.reshape(batch_size, seq_length,
                                    patch_height, patch_width,
                                    patch_size, patch_size,
                                    1)
            b = a.transpose(3, 4)
            img_tensor_i = b.reshape(batch_size, seq_length,
                                    patch_height * patch_size,
                                    patch_width * patch_size,
                                    1)
            
            img_tensor_list.append(img_tensor_i)

        img_tensor = torch.cat(img_tensor_list, -1)

    else:
        a = patch_tensor.reshape(batch_size, seq_length,
                                    patch_height, patch_width,
                                    patch_size, patch_size,
                                    img_channels)
        b = a.transpose(3, 4)
        img_tensor = b.reshape(batch_size, seq_length,
                                    patch_height * patch_size,
                                    patch_width * patch_size,
                                    img_channels)
    return img_tensor