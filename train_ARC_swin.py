# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import torch

from Model.Model_Config import CONFIGS

from Model.ARC_swin_b import swin_base_patch4_window7_224
from Utils.tools import count_parameters, AverageMeter

from tqdm import tqdm
from Utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule


from Utils.Frozen_weight import Swin_ARC_Frozen
from Data_process.VTAB_config import DATA_CONFIGS
from Data_process.VTAB_loader import get_data

logger = logging.getLogger(__name__)

def save_model(args, model):
    model_to_save = model.Module if hasattr(model, "module") else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup(args):
    data_cfg = DATA_CONFIGS[args.dataset]

    num_classes = data_cfg.Num_Classes
    model = swin_base_patch4_window7_224(num_classes=num_classes,pretrained=True)

    model.to(args.device)

    Swin_ARC_Frozen(model)

    num_params = count_parameters(model)
    # logger.info("{}".format(config))
    logger.info("Training HypeParameters %s", args)
    logger.info("Total Parameter: \t%2.2fM" % num_params)

    for name, para in model.named_parameters():
        if para.requires_grad == True:
            print(name)

    return model,num_params

def valid(args, model, test_loader):
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                      desc="Testing (X / X steps) (loss = X.X)",
                      bar_format='{l_bar}-{r_bar}',
                      dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, label = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, label)
            eval_losses.update(eval_loss.item())

            prediction = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(prediction.detach().cpu().numpy())
            all_label.append(label.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], prediction.detach().cpu().numpy(), axis=0)
            all_label[0] = np.append(
                all_label[0], label.detach().cpu().numpy(), axis=0)

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    # 计算平均准确率
    all_preds = all_preds[0]
    all_label = all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    return accuracy


def train(args, model):
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, test_loader = get_data(data_path=args.data_path, name=args.dataset, batch_size=args.train_batch_size)

    epoch_steps = len(train_loader)
    total_steps = epoch_steps * args.num_epochs
    warm_steps = epoch_steps * args.warmup_epochs
    evl_steps = args.eval_every * epoch_steps

    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)

    if args.decay_type == "cosine":
        schedule = WarmupCosineSchedule(optimizer, warmup_steps=warm_steps, t_total=total_steps)
    else:
        schedule = WarmupLinearSchedule(optimizer, warmup_steps=warm_steps, t_total=total_steps)

    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", total_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)


    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step, best_acc = 0, 0


    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                                desc="Training (X / X steps) (loss = X.X)",
                                bar_format='{l_bar}-{r_bar}',
                                dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, label = batch

            loss = model(x, label)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item())

                torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, total_steps, losses.val))

                if (global_step + 1) % evl_steps == 0:
                    accuracy = valid(args, model, test_loader)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

            if (global_step + 1) % total_steps == 0:
                break

        losses.reset()
        if (global_step + 1) % total_steps == 0:
            break

    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    return best_acc

def main(config=None):
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_path",  default= "/root/autodl-tmp/VTAB_Data", help="where is the data store, should be changed according to local file")

    parser.add_argument("--name", default="ViT-B_16_cifar_att_mlp_independent_50dim_bias0_repadapterStructure_error",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["caltech101", "cifar", "smallnorb_ele", "sun397", "svhn", "resisc45", "dmlab"
                                              "oxford_iiit_pet","oxford_flowers102", 'dtd', "kitti", "eurosat","clevr_count"
                                              ,"dsprites_ori"],
                        default= "kitti",  help="Which downstream task.")


    parser.add_argument("--tuning_mode", choices=["ARC_att", "ARC"],
                        default= "ARC",  help="tuning mode,can be ARC_att or ARC")

    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-L_16", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=256, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=0.01, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--num_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--vit_drop", default=0, type=float,
                        help="dropout rate of ViT.")
    parser.add_argument("--adapt_drop", default=0.1, type=float,
                        help="dropout rate of adapter.")

    args = parser.parse_args()


    if config:
        args.dataset = config["dataset"]
        name = "SwinB_VTAB_dim50_ARCatt_" + args.dataset + "_lr%f" % config["lr"] + "_wd%f" % config["wd"] + "_vdrop0.0_adrop%f" %  config["a_drop"]
        args.name = name
        args.learning_rate = config["lr"]
        args.weight_decay = config["wd"]
        args.a_drop = config["a_drop"]


    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("Process device: %s, n_gpu: %s" % (args.device, args.n_gpu))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    model, para_number = setup(args)

    # Training
    best_acc = train(args, model)
    
    
    return best_acc, para_number

if __name__ == "__main__":
    config_list = []

    cifar_config ={"dataset" : "cifar",
        "lr" : 0.005,
        "wd" : 0.01,
        "a_drop" : 0.1}
    config_list.append(cifar_config)
    
    caltech101_config ={"dataset" : "caltech101",
        "lr" : 0.003,
        "wd" : 0.05,
        "a_drop" : 0.1}
    config_list.append(caltech101_config)
    
    dtd_config ={"dataset" : "dtd",
        "lr" : 5e-3,
        "wd" : 5e-2,
        "a_drop" : 0.8}
    config_list.append(dtd_config)
    
    oxford_flowers102_config ={"dataset" : "oxford_flowers102",
        "lr" : 0.005,
        "wd" : 0.00005,
        "a_drop" : 0.5}
    config_list.append(oxford_flowers102_config)
    
    oxford_iiit_pet_config ={"dataset" : "oxford_iiit_pet",
        "lr" : 0.01,
        "wd" : 0.05,
        "a_drop" : 0.1}
    config_list.append(oxford_iiit_pet_config)
    
    svhn_config ={"dataset" : "svhn",
        "lr" : 0.01,
        "wd" : 0.05,
        "a_drop" : 0.1}
    config_list.append(svhn_config)
    
    sun397_config ={"dataset" : "sun397",
        "lr" : 0.005,
        "wd" : 0.00005,
        "a_drop" : 0.1}
    config_list.append(sun397_config)
    
    eurosat_config ={"dataset" : "eurosat",
        "lr" : 0.003,
        "wd" : 0,
        "a_drop" : 0.1}
    config_list.append(eurosat_config)
    
    resisc45_config ={"dataset" : "resisc45",
        "lr" : 0.01,
        "wd" : 0.0,
        "a_drop" : 0.5}
    config_list.append(resisc45_config)
    
    patch_camelyon_config ={"dataset" : "patch_camelyon",
        "lr" : 0.005,
        "wd" : 0.00005,
        "a_drop" : 0.1}
    config_list.append(patch_camelyon_config)
    
    print(os.path.exists("Train/output"))
    
    diabetic_retinopathy_config ={"dataset" : "diabetic_retinopathy",
        "lr" : 0.005,
        "wd" : 0.00005,
        "a_drop" : 0.1}
    config_list.append(diabetic_retinopathy_config)

    clevr_count_config ={"dataset" : "clevr_count",
        "lr" : 0.002,
        "wd" : 0.05,
        "a_drop" : 0.5}
    config_list.append(clevr_count_config)

    clevr_dist_config ={"dataset" : "clevr_dist",
        "lr" : 0.001,
        "wd" : 0.05,
        "a_drop" : 0.1}
    config_list.append(clevr_dist_config)

    dmlab_config ={"dataset" : "dmlab",
        "lr" : 0.005,
        "wd" : 0.01,
        "a_drop" : 0.1}
    config_list.append(dmlab_config)

    kitti_config ={"dataset" : "kitti",
        "lr" : 0.01,
        "wd" : 0.0,
        "a_drop" : 0.1}
    config_list.append(kitti_config)

    smallnorb_azi_config ={"dataset" : "smallnorb_azi",
        "lr" : 0.01,
        "wd" : 0.005,
        "a_drop" : 0.1}
    config_list.append(smallnorb_azi_config)

    smallnorb_ele_config ={"dataset" : "smallnorb_ele",
        "lr" : 0.001,
        "wd" : 0.05,
        "a_drop" : 0.1}
    config_list.append(smallnorb_ele_config)

    dsprites_loc_config ={"dataset" : "dsprites_loc",
        "lr" : 0.01,
        "wd" : 0.0,
        "a_drop" : 0.1}
    config_list.append(dsprites_loc_config)

    dsprites_ori_config ={"dataset" : "dsprites_ori",
        "lr" : 0.005,
        "wd" : 0.0,
        "a_drop" : 0.5}
    config_list.append(dsprites_ori_config)

    results_list = []

    for config in config_list:
        print(config)
        result = {}
        result["dataset"] = config["dataset"]
        acc, para_num = main(config)
        result["acc"] = acc
        result["para_num"] = para_num
        results_list.append(result)
        
    count = 0
    for result in results_list:
        print(result)
        count = count + result["para_num"]
        
    print(count/19)