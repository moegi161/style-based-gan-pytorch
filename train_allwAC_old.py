import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils, models

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult

# function for load the correct classifier of specified resolution
def load_attr_classifier(path, image_size):
    model_path = path + '/attr_classifier_' + str(image_size) + '.pth'
    model_dict = torch.load(model_path)
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=40)
    model.load_state_dict(model_dict['model_state_dict'])
    model = nn.DataParallel(model).cuda()
    model.eval()

    return model

def train(args, dataset, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2 # init size = 8 -> step = 1
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)
    if resolution <= 512:  # (not used at 1024x1024) / alpha == 1 (only use attr classifier on stabilization stage?)
        classifier = load_attr_classifier(args.attr_classifier, resolution)

    criterion = nn.BCEWithLogitsLoss()
    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(args.i,3_000_000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0
    fake_attr_loss_val = 0
    real_attr_loss_val = 0
    attr_loss_val = 0

    alpha = 0
    used_sample = args.batch.get(resolution, args.batch_default) * args.i
    #print(used_sample)

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in pbar:
        # Update discrimintor parameter
        discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)
            if resolution <= 512: # (not used at 1024x1024) / alpha == 1 (only use attr classifier on stabilization stage?)
                classifier = load_attr_classifier(args.attr_classifier, resolution)

            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_aac_hd_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image, labels = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, labels = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        #print(b_size, labels.shape)
        real_image = real_image.cuda()
        labels = labels.cuda()

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = F.softplus(-real_scores).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_scores.sum(), inputs=real_image, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9: #style mixing?
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        #print(len(gen_in1),len(gen_in1[0]), len(gen_in1[1]), labels.shape)
        fake_image = generator(gen_in1, labels, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            if i%10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()

        # Update Generator Parameter
        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, labels, step=step, alpha=alpha)
            predict = discriminator(fake_image, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = -predict.mean() # G_wgan loss in official impl

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean() # G_logistic_ns loss in official impl

            if i%10 == 0:
                gen_loss_val = loss.item()

            if resolution <= 512:
                real_attr = classifier(real_image)
                fake_attr = classifier(fake_image)
                ## loss from atrribute classifier
                fake_attr_loss = criterion(fake_attr, labels)
                real_attr_loss = criterion(real_attr, labels)
                attr_loss = fake_attr_loss - real_attr_loss
                loss += attr_loss
                if i % 10 == 0:
                    fake_attr_loss_val = fake_attr_loss.item()
                    real_attr_loss_val = real_attr_loss.item()
                    attr_loss_val = attr_loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))

            with torch.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            torch.randn(gen_j, code_size).cuda(),labels.cuda(), step=step, alpha=alpha
                        ).data.cpu()
                    )

            utils.save_image(
                torch.cat(images, 0),
                f'sample/aac_hd_{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 5000 == 0:
            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_aac_hd-{str(i + 1).zfill(6)}.model',
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f}; fake_AC: {fake_attr_loss_val:.3f}; '
            f'real_AC: {real_attr_loss_val:.3f}; AC:{attr_loss_val:.3f}'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('--img_path', type=str, help='path of specified dataset')
    parser.add_argument('--label_path', type=str, help='path of label of dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=600_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument(
        '--attr_classifier',
        default='./attr_classifier',
        type=str,
        help='path of folder for attribute classifier')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument('--i', default=0, type=int, help='resume step')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

    args = parser.parse_args()

    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()
    g_running = StyledGenerator(code_size).cuda()
    g_running.train(False)

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator.module, 0)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)

        generator.module.load_state_dict(ckpt['generator'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True),
        ]
    )

    # get the label of images
    label_list = open(args.label_path).readlines()[1:]
    data_label = []
    for i in range(len(label_list)):
        data_label.append(label_list[i].split())
    print(len(label_list))

    # transform label into 0 and 1
    for m in range(len(data_label)):
        data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
        data_label[m] = [int(p) for p in data_label[m]]

    #print(len(data_label))
    #print(data_label[0])

    dataset = MultiResolutionDataset(args.img_path, data_label, transform)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.001, 1024: 0.001}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    train(args, dataset, generator, discriminator)
