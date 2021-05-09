import argparse
import math
import os

import torch
from torch import nn
from torchvision import utils, models

from model import StyledGenerator


@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        #style = generator.mean_style(torch.randn(1024, 552).to(device)) #zl
        style = generator.mean_style(torch.randn(1024, 512).to(device))  # wl

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

@torch.no_grad()
def sample(generator,labels, step, mean_style, n_sample, device):
    image = generator(
        torch.randn(n_sample, 512).to(device),
        labels.to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    
    return image

@torch.no_grad()
def style_mixing(generator, labels, step, mean_style, n_source, n_target, device):
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    source_image = generator(
        source_code, labels.to(device), step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, label.to(device), step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            labels.to(device),
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)
    
    return images

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--num', type=int, default=20259, help='total num of test image')
    parser.add_argument('--n_row', type=int, default=4, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=6, help='number of columns of sample matrix')
    parser.add_argument('path', type=str, help='path to checkpoint file')
    parser.add_argument('--label', type=str, help='path to input labels')
    parser.add_argument('--seed', type=str, default=5, help='seed')
    
    args = parser.parse_args()
    result_dir = "result/" + args.path.split("/")[1].split(".")[0]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(result_dir+"/samples"):
        os.makedirs(result_dir+"/samples")
    
    device = 'cuda'
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path)['g_running'])
    generator.eval()

    mean_style = get_mean_style(generator, device)

    # get the label of images
    label_list = open(args.label, encoding='utf-16').readlines()[1:]
    #label_list = open(args.label).readlines()[1:]
    data_label = []
    for i in range(len(label_list)):
        data_label.append(label_list[i].split())
    print(len(data_label))
    for m in range(len(data_label)):
        data_label[m] = [n.replace('-1', '0') for n in data_label[m]][1:]
        data_label[m] = [int(p) for p in data_label[m]]
    label = torch.Tensor(data_label) #in terms  of [0,1]

    step = int(math.log(args.size, 2)) - 2

    checkpoint_path = './attr_classifier'
    if args.size < 1024:
        model = load_attr_classifier(checkpoint_path, args.size)
    else:
        model = load_attr_classifier(checkpoint_path, 512)

    result_path = result_dir + '/attr_pred.txt'
    f = open(result_path, "w")
    #args.label = "./data/label/list_attr_celeba_test.txt"
    attributes = open(args.label, encoding='utf-16').readlines()[0].split()
    for t in range(len(attributes)):
        f.write("%s " % attributes[t])
    f.write("\n")
    id = 1
    for j in range(0,args.num,args.n_row * args.n_col):
        if j < args.num - args.n_row * args.n_col:
            img = sample(generator, label[j:j+(args.n_row * args.n_col)], step, mean_style, args.n_row * args.n_col, device)
        else:
            img = sample(generator, label[j:], step, mean_style, args.num - j, device)
        utils.save_image(img[:(args.n_row * args.n_col)], result_dir + '/sample_' + str(j) + '.png', nrow=args.n_col,
                         normalize=True, range=(-1, 2), scale_each=True)


        for i in img:
            #f.write('{:06d}'.format(id) + '.png ')
            # try:
            # image_set = Image.open('./{:05d}'.format(id) + '.png')
            # f.write('{:05d}'.format(id) + '.png ')
            #print(id)
            utils.save_image(i, result_dir + '/samples/{:06d}.png'.format(id), normalize=True, range=(-1, 2))
            image = torch.unsqueeze(i, 0)
            #image = image.to('cuda')
            output = model(image)
            #print(output)
            sig = nn.Sigmoid()
            result = sig(output)

            result = result.cpu().detach().numpy()
            #print(result)
            for t in range(len(attributes)):
                f.write("%s " % (result[0][t]))
            f.write("\n")
            id += 1

        if j % 1000 == 0:
            print(j)




    '''
    for j in range(20):
        img = style_mixing(generator, label, step, mean_style, args.n_col, args.n_row, device)
        utils.save_image(
            img, f'result/aac/sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
        )
    '''