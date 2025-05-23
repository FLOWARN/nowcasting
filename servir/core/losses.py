import numpy as np
import torch 


def CFSSSurrogateLoss(gt, pred, avg_kernel_half_width, threshold, g_func):
    """
    Complementary FSS Surrogate Loss
    This function computes an upper bound to the complementary FSS (Fraction Skill Score) 
    and outputs the loss value as a torch variable.
    Inputs:
        gt: (torch.tensor) ground truth image sequence of shape (batch size, time_instances,  image_length, image_width)
        pred: (torch.tensor) predicted image sequence of shape (batch size, time_instances, image_length, image_width)
        avg_kernel_half_width: (int) value used to construct the averaging kernel (0 <= n < (min(image_length,image_width)/2))
        threshold: (float64) threshold used for FSS calculations (for "significant" precipitation)
    """
    # averaging kernel with no padding
    averaging_kernel = torch.nn.AvgPool2d(kernel_size=(2*avg_kernel_half_width+1, 2*avg_kernel_half_width+1), stride=1)
    
    # g(x, \hat{x} )
    gt_thresholded = gt - torch.ones_like(gt)*threshold
    pred_thresholded = pred - torch.ones_like(pred)*threshold
    
    prod_gt_pred_thresholded = g_func(torch.mul(gt_thresholded, pred_thresholded))
    
    return averaging_kernel(prod_gt_pred_thresholded).mean()

    
def g_func_relu(tensor):
    return torch.relu(torch.ones_like(tensor) - tensor)

def g_func_log(tensor):
    return torch.log(torch.ones_like(tensor) + torch.exp(-tensor))


# testing FSS surrogate loss with random inputs 
def main():
    batch_size = 32
    time_instances = 12
    gt = np.random.random((batch_size,time_instances ,80,70))
    gt = torch.from_numpy(gt)
    pred = np.random.random((batch_size,time_instances ,80,70))
    pred = torch.from_numpy(pred)   


    avg_kernel_half_width = 5
    threshold = 8/60.0

    g_func = g_func_log

    loss_value = CFSSSurrogateLoss(gt, pred, avg_kernel_half_width, threshold, g_func)

    print(loss_value)
    
if __name__ == "__main__":
    main()
