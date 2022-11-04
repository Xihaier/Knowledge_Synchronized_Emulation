import argparse
import warnings

from data import getData
from models import get_model
from train import Trainer
from train import get_loss, get_optimizer, get_scheduler
from utils import set_seed, set_device, set_logger

# Ignore warning
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser(description='Deep Finite Difference Emulator --- Burgers Equation')
    # fundamental
    parser.add_argument("--randomseed", type=int, default=1225)
    parser.add_argument("--local_rank", type=int, default=0, help="For distributed training: local_rank")
    
    # data
    parser.add_argument("--test_data_dir", type=str, default='./data/data_64/')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ntrain", type=int, default=256)
    parser.add_argument("--ntest", type=int, default=512)
    parser.add_argument("--btrain", type=int, default=128)
    parser.add_argument("--nel", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--test_every", type=int, default=2)
    parser.add_argument("--test_step", type=int, default=200)
    parser.add_argument("--nu", type=float, default=0.005)

    # model
    parser.add_argument("--model", type=str, default="FNO", choices=["FNO", "PhyFNO", "ResNet", "PhyResNet"])
    parser.add_argument("--input_channels", type=int, default=6)
    parser.add_argument("--output_channels", type=int, default=2)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout_rate", type=float, default=0.)

    # optim
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--optim_loss", type=str, default="LogCoshLoss", choices=["LogCoshLoss", "XTanhLoss", "XSigmoidLoss", "MSELoss"])
    parser.add_argument("--optim_alg", type=str, default="AdamW", choices=["Adam","AdamW","AdamL2"])
    parser.add_argument("--optim_wd", type=float, default=1e-4)
    parser.add_argument("--optim_lr", type=float, default=0.001)
    parser.add_argument("--optim_lr_init", type=float, default=1e-5)
    parser.add_argument("--optim_lr_final", type=float, default=1e-5)
    parser.add_argument("--optim_scheduler", type=str, default="ExponentialLR", choices=["Cosine", "SquareRoot", "StepLR", "ExponentialLR", "ReduceLROnPlateau"])
    parser.add_argument("--optim_warmup", type=int, default = 50)    
    
    # log
    parser.add_argument("--logroot", type=str, default='./logs/')
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--ckpt", type=bool, default=False)
    parser.add_argument("--ckpt_epoch", type=int, default=None)    
    args = parser.parse_args()
    return args 


if __name__ == '__main__':
    ## get argument parser
    args = get_parser()

    ## set logging
    set_logger(args)

    ## get device
    set_device(args)

    ## set random seed
    set_seed(args)

    ## get dateset and loader
    train_loader, test_loader = getData(args)

    ## get model
    model = get_model(args)
    model = model.to(args.device)

    ## get loss
    criterion = get_loss(args)

    ## get optimizer
    optimizer = get_optimizer(model, args)

    ## get scheduler
    scheduler = get_scheduler(args, optimizer)

    # get trainer module
    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, args=args)

    for epoch in range(args.epochs):
        trainer.train_step(epoch, train_loader)
        trainer.eval_step(epoch, test_loader)
        trainer.update_lr(epoch)

