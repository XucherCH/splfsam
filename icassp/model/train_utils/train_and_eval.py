import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import model.train_utils.distributed_utils as utils


def LossFunc(pred, mask):

    BCEWL = F.binary_cross_entropy_with_logits
    
    bce = BCEWL(pred,mask)



    #mae = F.l1(pred, mask, reduce=None)

    return bce 


def evaluate(model, data_loader, device):
    model.eval()
    mae_metric = utils.MeanAbsoluteError()
    f1_metric = utils.F1Score()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images, targets = images.to(device), targets.to(device)
            output = model(images)[-1]

            # post normalization
            # ma = torch.max(output)
            # mi = torch.min(output)
            # output = (output - mi) / (ma - mi)

            mae_metric.update(output, targets)
            f1_metric.update(output, targets)

        mae_metric.gather_from_all_processes()
        f1_metric.reduce_from_all_processes()

    return mae_metric, f1_metric


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.amp.autocast(enabled=scaler is not None,device_type="cuda"):
            output = model(image)

            loss = LossFunc(output[0],target)+LossFunc(output[1],target)+LossFunc(output[2],target)+LossFunc(output[3],target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        Return a learning rate multiplier factor based on the step number,
        note that pytorch will call the lr_scheduler.step() method in advance before training starts
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # During warmup, the lr multiplier factor changes from warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # After warmup, the lr multiplier factor changes from 1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group
