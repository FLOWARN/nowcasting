import time
from decimal import Decimal
from tqdm import tqdm
from typing import Dict, List, Union
 
import numpy as np
import torch
import torch.nn as nn
from timm.utils.agc import adaptive_clip_grad
from timm.utils import AverageMeter
from contextlib import suppress

from servir.core.optimizor import get_optim_scheduler
from servir.utils.convLSTM_utils import reshape_patch, reshape_patch_back, schedule_sampling, reserve_schedule_sampling_exp
from servir.utils.distributed_utils import reduce_tensor
from servir.core.losses import CFSSSurrogateLoss, g_func_relu, g_func_log 



class ConvLSTMCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, height, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, height, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size,
                          stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t):
        
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * g_t
        return h_new, c_new



class ConvLSTM_Model(nn.Module):
    r"""ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    """

    def __init__(self, num_layers, num_hidden, config, **kwargs):
        super(ConvLSTM_Model, self).__init__()
        if 'input_channel' in config:
            C_in = config['input_channel']
        else:
            C_in = config['channels']

        H, W = config['img_height'], config['img_width']

        self.config = config
        self.frame_channel = config['patch_size'] * config['patch_size'] * C_in
        self.output_channel = config['patch_size'] * config['patch_size'] * config['channels']
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // config['patch_size']
        width = W // config['patch_size']

        if config['loss'] == 'MSE':
            self.criterion = nn.MSELoss()
        elif config['loss'] == 'CFSSS':
            self.criterion = CFSSSurrogateLoss

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, config['filter_size'],
                                       config['stride'], config['layer_norm'])
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.output_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        

    def forward(self, frames_tensor, mask_true, **kwargs):

        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.config['device'])
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.config['in_seq_length'] + self.config['out_seq_length'] - 1):
            # reverse schedule sampling
            if self.config['reverse_scheduled_sampling'] == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.config['in_seq_length']:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.config['in_seq_length']] * frames[:, t] + \
                        (1 - mask_true[:, t - self.config['in_seq_length']]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])

            if self.config['relu_last']:
                x_gen = torch.relu(x_gen)

            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()

        # if kwargs.get('skip_frame_loss')==True:
        #     loss = self.MSE_criterion(next_frames[:, -self.config['out_seq_length']::2],\
        #                             frames_tensor[:, -self.config['out_seq_length']::2, :, :, :input_channel_num])

        # if self.config['loss'] == 'CFSSS':
        #     img_frames_tensor = frames_tensor[:, :, :, :, :input_channel_num]
        #     next_frames_prefix = torch.cat([img_frames_tensor[:, 0:1], next_frames], dim=1)
        #     next_frames_img = reshape_patch_back(next_frames_prefix, self.config['patch_size'], self.config['channel_sep'])
        #     frames_tensor_img = reshape_patch_back(img_frames_tensor, self.config['patch_size'],self.config['channel_sep'])
        #     loss = self.criterion(next_frames_img[:, :, :, :, 0],\
        #                         frames_tensor_img[:, :, :, :, 0],\
        #                         0, 1.0/60.0, g_func_log)

        if kwargs.get('return_loss')==True:
            if 'loss_channels' in kwargs:
                reshaped_channel_num = self.config['patch_size'] ** 2 * int(kwargs.get('loss_channels'))
                loss = self.criterion(next_frames[:, :, :, :, :reshaped_channel_num], frames_tensor[:, 1:, :, :, :reshaped_channel_num])
            else: # all losses
                loss = self.criterion(next_frames, frames_tensor[:, 1:, :, :, :])
        else:
            loss = None

        return next_frames, loss


class ConvLSTM():
    """ConvLSTM

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.

    Notice: ConvLSTM requires `find_unused_parameters=True` for DDP training.
    """

    def __init__(self, config):
            
        self.device = config['device']
        self.dist = config['distributed']
        self.config = config

        self.model = self._build_model()

        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer()

        if self.config['loss'] == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.config['loss'] == 'CFSSS':
            self.criterion = CFSSSurrogateLoss


        self.clip_value = self.config['clip_grad']
        self.clip_mode = config['clip_mode'] if self.clip_value is not None else None

        # # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing

        self.loss_scaler = None
        # # setup metrics

        # self.metric_list = ['mse', 'mae']


    def _build_model(self):
        num_hidden = [int(x) for x in self.config['num_hidden'].split(',')]
        num_layers = len(num_hidden)


        model = ConvLSTM_Model(num_layers, num_hidden, self.config)

        return model.to(self.device)
    
    def _init_optimizer(self):
        # epochs = min(self.config['max_epoch'], self.config['early_stop_epoch'])
        epochs = self.config['max_epoch']
        return get_optim_scheduler(self.config, self.model, epochs)

    
    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        channel_sep = kwargs.get('channel_sep')
        #=== img order [S, T, C, H, W] --> [S, T, H, W, C]
        test_ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        #=== img reshpe [S, T, H, W, C] --> [S, T, H//patch_size, W//patch_size, patch_size**2*C]
        test_dat = reshape_patch(test_ims, self.config['patch_size'], channel_sep=channel_sep)

        # reverse schedule sampling
        if self.config['reverse_scheduled_sampling'] == 1:
            mask_input = 1
        else:
            mask_input = self.config['out_seq_length']

        img_channel, img_height, img_width = self.config['channels'], self.config['img_height'], self.config['img_width']


        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            test_dat.shape[1] - mask_input - 1,
            img_height // self.config['patch_size'],
            img_width // self.config['patch_size'],
            self.config['patch_size'] ** 2 * img_channel)).to(self.device)
            
        if self.config['reverse_scheduled_sampling'] == 1:
            real_input_flag[:, :self.config['out_seq_length'] - 1, :, :] = 1.0

        img_gen, _ = self.model(test_dat, real_input_flag, return_loss=False)
        img_gen = reshape_patch_back(img_gen, self.config['patch_size'], channel_sep=channel_sep)   
        pred_y = img_gen[:, -self.config['out_seq_length']:].permute(0, 1, 4, 2, 3).contiguous()

        return pred_y

    def _collect_evaluate_predictions(self, data_loader, setName, gather_pred, withMeta, **kwargs):
        """Evaluate the model with val_loader.

        Args:
            data_loader: dataloader of vali/test/train set.
            setName: name of the set. validation or test set
            gather_pred: whether to gather predictions. 
            withMeta: whether to return meta data. 

        Returns:
            list(tensor, ...): The list of predictions and losses.
            eval_log(str): The string of metrics.
        """
        self.model.eval()

        data_loss = AverageMeter()
        data_time_m = AverageMeter()
        pbar = tqdm(data_loader)

        # loop
        pred_results = []
        pred_meta = []
        for batch in pbar:

            if withMeta:
                batch_x, batch_y, batch_x_dt, batch_y_dt = batch

            else:
                batch_x, batch_y = batch

            channel_sep = kwargs.get('channel_sep') 
            st = time.time()
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device, dtype=torch.float32), batch_y.to(self.device, dtype=torch.float32)
                pred_y = self._predict(batch_x, batch_y, channel_sep=channel_sep)
                # self.config['channels']
                # if skip_frame_loss:
                #     loss = self.criterion(pred_y[:, -self.config['out_seq_length']::2],\
                #                         batch_y[:, -self.config['out_seq_length']::2]).cpu().numpy().item()

                # if self.config['loss'] == 'CFSSS':
                #     img_batch_y = batch_y[:, :, 0, :, :]
                #     img_pred_y = pred_y[:, :, 0, :, :]
                #     loss = CFSSSurrogateLoss(img_pred_y, img_batch_y,0, 1.0/60.0, g_func_log).cpu().numpy().item()
                if 'loss_channels' in kwargs:
                    loss_channels = kwargs.get('loss_channels')
                    loss = self.criterion(pred_y[:, :, 0:loss_channels, :, :], batch_y[:, :, 0:loss_channels, :, :]).cpu().numpy().item()
                else:
                    loss = self.criterion(pred_y, batch_y).cpu().numpy().item()
                    
                data_loss.update(loss, batch_x.size(0)) 
                data_time_m.update(time.time() - st)

                if self.config['rank'] == 0:
                    log_buffer = "{} loss : {:.2E}".format(setName, Decimal(loss)) #'{} loss: {:.4f}'.format(setName,loss)
                    log_buffer += ' | avg {} time: {:.4f}'.format(setName, data_time_m.avg)
                    pbar.set_description(log_buffer)
                
                if gather_pred==True:
                    preds = pred_y.cpu().numpy()

                    if preds.shape[2] == 1: # if grayscale, remove the channel dimension. [S, T, 1, H, W] --> [S, T, H, W]
                        preds = np.squeeze(preds, axis=2)

                    pred_results.append(preds)

                    if withMeta:
                        pred_meta = pred_meta + [dt for dt in batch_y_dt]

        if len(pred_results)>0:
            pred_results = np.concatenate(pred_results, axis=0)

        return data_loss.avg, pred_results, pred_meta
    
    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.model_optim, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.model_optim.param_groups]
        elif isinstance(self.model_optim, dict):
            lr = dict()
            for name, optim in self.model_optim.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def clip_grads(self, params, norm_type: float = 2.0):
        """ Dispatch to gradient clipping method

        Args:
            parameters (Iterable): model parameters to clip
            value (float): clipping value/factor/norm, mode dependant
            mode (str): clipping mode, one of 'norm', 'value', 'agc'
            norm_type (float): p-norm, default 2.0
        """
        if self.clip_mode is None:
            return
        if self.clip_mode == 'norm':
            torch.nn.utils.clip_grad_norm_(params, self.clip_value, norm_type=norm_type)
        elif self.clip_mode == 'value':
            torch.nn.utils.clip_grad_value_(params, self.clip_value)
        elif self.clip_mode == 'agc':
            adaptive_clip_grad(params, self.clip_value, norm_type=norm_type)
        else:
            assert False, f"Unknown clip mode ({self.clip_mode})."

    def train_one_epoch(self, train_loader, epoch, num_updates, eta=None, return_loss = True, channel_sep=False, loss_channels=1):

        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        # training mode
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.config['rank'] == 0 else train_loader

        
        for batch_x, batch_y in train_pbar:
            st = time.time()    
            
            self.model_optim.zero_grad()
            # send data to gpu
            batch_x, batch_y = batch_x.to(self.device, dtype=torch.float32), batch_y.to(self.device, dtype=torch.float32)

            # preprocess
            #=== img order [S, T, C, H, W] --> [S, T, H, W, C]
            ims = torch.cat([batch_x, batch_y], dim=1).permute(0, 1, 3, 4, 2).contiguous()
            #=== img reshpe [S, T, H, W, C] --> [S, T, H//patch_size, W//patch_size, patch_size**2*C]
            ims = reshape_patch(ims, self.config['patch_size'], channel_sep=channel_sep)
            
            if self.config['reverse_scheduled_sampling'] == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.config)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.config)

            with self.amp_autocast():
                _, loss = self.model(ims, real_input_flag, return_loss=return_loss, loss_channels=loss_channels)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.config['clip_grad'], clip_mode=self.config['clip_mode'],
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            
            data_time_m.update(time.time() - st)

            if self.config['rank'] == 0:
                log_buffer = "train loss : {:.2E}".format(Decimal(loss.item())) #'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | avg train time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)


        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m.avg, eta

    def vali(self, data_loader, gather_pred=False,  channel_sep=False, loss_channels=1):
        """Evaluate the model with test_loader.

        Args:
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        vali_loss, _, _ = self._collect_evaluate_predictions(data_loader, gather_pred=gather_pred,\
                                                            setName='vali', withMeta=False,\
                                                            channel_sep=channel_sep, loss_channels=loss_channels)

        return vali_loss


    def test(self, data_loader, gather_pred=True, channel_sep=True, loss_channels=1):
        """Evaluate the model with test_loader.

        Args:
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        test_loss, test_results, test_meta = self._collect_evaluate_predictions(data_loader, gather_pred=gather_pred,\
                                                                                setName='test', withMeta=True, \
                                                                                channel_sep=channel_sep, loss_channels=loss_channels)

        return test_loss, test_results, test_meta 


