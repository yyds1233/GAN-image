# Adversarial Attacks with AdvGAN and AdvRaGAN
# Copyright(C) 2020 Georgios (Giorgos) Karantonis
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

### 404 航母
### 577 小船
### 629 轮船
### 这个数据集的标签比黑白盒的 imagenet 索引多 1

import os
import json

import numpy as np

import torch
import torchvision.datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.models as m

import models
import custom_data as cd
from advGAN import AdvGAN_Attack


def load_hyperparameters(config_file):
    with open(config_file) as hp_file:
        hyperparams = json.load(hp_file)

    target = hyperparams['target_dataset']
    lr_target = hyperparams['target_learning_rate']
    epochs_target = hyperparams['target_model_epochs']
    l_inf_bound = hyperparams['maximum_perturbation_allowed']

    epochs = hyperparams['AdvGAN_epochs']
    lr = hyperparams['AdvGAN_learning_rate']
    alpha = hyperparams['alpha']
    beta = hyperparams['beta']
    gamma = hyperparams['gamma']
    kappa = hyperparams['kappa']
    c = hyperparams['c']
    n_steps_D = hyperparams['D_number_of_steps_per_batch']
    n_steps_G = hyperparams['G_number_of_steps_per_batch']
    is_relativistic = True if hyperparams['is_relativistic'] == 'True' else False

    return target, lr_target, epochs_target, l_inf_bound, epochs, lr, alpha, beta, gamma, kappa, c, n_steps_D, n_steps_G, is_relativistic


def create_dirs():
    if not os.path.exists('./results/examples/MNIST/train/'):
        os.makedirs('./results/examples/MNIST/train/')

    if not os.path.exists('./results/examples/CIFAR10/train/'):
        os.makedirs('./results/examples/CIFAR10/train/')

    if not os.path.exists('./results/examples/HighResolution/train/'):
        os.makedirs('./results/examples/HighResolution/train/')

    if not os.path.exists('./checkpoints/target/'):
        os.makedirs('./checkpoints/target/')

    if not os.path.exists('./npy/MNIST/'):
        os.makedirs('./npy/MNIST/')

    if not os.path.exists('./npy/CIFAR10/'):
        os.makedirs('./npy/CIFAR10/')

    if not os.path.exists('./npy/HighResolution/'):
        os.makedirs('./npy/HighResolution/')


def init_params(target):
    if target == 'MNIST':
        batch_size = 128
        l_inf_bound = 0.3 if L_INF_BOUND == 'Auto' else float(L_INF_BOUND)

        n_labels = 10
        n_channels = 1

        target_model = models.MNIST_target_net().to(device)

        dataset = torchvision.datasets.MNIST(
            './datasets',
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    elif target == 'CIFAR10':
        batch_size = 400
        l_inf_bound = 8 / 255 if L_INF_BOUND == 'Auto' else float(L_INF_BOUND) / 255

        n_labels = 10
        n_channels = 3

        target_model = models.resnet32().to(device)

        dataset = torchvision.datasets.CIFAR10(
            './datasets',
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    elif target == 'HighResolution':
        batch_size = 30
        l_inf_bound = 0.01 if L_INF_BOUND == 'Auto' else float(L_INF_BOUND)

        n_labels = 1000
        n_channels = 3

        target_model = m.inception_v3(weights='DEFAULT').to(device)
        target_model.eval()

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        dataset = cd.HighResolutionDataset('/app/src/test_dataset/img', transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    else:
        raise NotImplementedError('Unknown Dataset')

    return dataloader, target_model, batch_size, l_inf_bound, n_labels, n_channels, len(dataset)


def train_target_model(target, target_model, epochs, dataloader, dataset_size):
    target_model.train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=LR_TARGET_MODEL)

    for epoch in range(epochs):
        loss_epoch = 0

        for i, data in enumerate(dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)

            logits_model = target_model(train_imgs)
            criterion = F.cross_entropy(logits_model, train_labels)
            loss_epoch += criterion

            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()

        print('Loss in epoch {}: {}'.format(epoch, loss_epoch.item()))

    targeted_model_file_name = './checkpoints/target/{}_bs_{}_lbound_{}.pth'.format(
        target, batch_size, l_inf_bound
    )
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()

    n_correct = 0
    for i, data in enumerate(dataloader, 0):
        img, label = data
        img, label = img.to(device), label.to(device)

        pred_lab = torch.argmax(target_model(img), 1)
        n_correct += torch.sum(pred_lab == label, 0)

    print('{} full dataset:'.format(target))
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in {} full dataset: {}%\n'.format(target, 100 * n_correct.item() / dataset_size))


def test_clean_performance(target, dataloader, target_model, dataset_size, mode='train'):
    n_correct = 0
    true_labels, pred_labels = [], []

    for i, data in enumerate(dataloader, 0):
        img, true_label = data
        img, true_label = img.to(device), true_label.to(device)

        pred_label = torch.argmax(target_model(img), 1)
        n_correct += torch.sum(pred_label == true_label, 0)

        true_labels.append(true_label.detach().cpu().numpy())
        pred_labels.append(pred_label.detach().cpu().numpy())

    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)

    np.save('./npy/{}/{}_clean_true_labels'.format(target, mode), true_labels)
    np.save('./npy/{}/{}_clean_pred_labels'.format(target, mode), pred_labels)

    print(target)
    print('Clean Correctly Classified in full dataset: {}'.format(n_correct.item()))
    print('Clean Accuracy in {} full dataset: {}%\n'.format(
        target, 100 * n_correct.item() / dataset_size)
    )

    return true_labels, pred_labels, n_correct.item()


def test_attack_performance(target, dataloader, mode, adv_GAN, target_model, batch_size, l_inf_bound, dataset_size):
    n_correct = 0

    true_labels, pred_labels = [], []
    img_np, adv_img_np = [], []

    for i, data in enumerate(dataloader, 0):
        img, true_label = data
        img, true_label = img.to(device), true_label.to(device)

        perturbation = adv_GAN(img)
        if perturbation.shape[-2:] != img.shape[-2:]:
            perturbation = F.interpolate(
                perturbation,
                size=img.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        adv_img = torch.clamp(perturbation, -l_inf_bound, l_inf_bound) + img
        adv_img = torch.clamp(adv_img, 0, 1)

        pred_label = torch.argmax(target_model(adv_img), 1)
        n_correct += torch.sum(pred_label == true_label, 0)

        true_labels.append(true_label.detach().cpu().numpy())
        pred_labels.append(pred_label.detach().cpu().numpy())
        img_np.append(img.detach().permute(0, 2, 3, 1).cpu().numpy())
        adv_img_np.append(adv_img.detach().permute(0, 2, 3, 1).cpu().numpy())

        true_label_cpu = true_label.detach().cpu().numpy()
        pred_label_cpu = pred_label.detach().cpu().numpy()

        print('Saving images for batch {} out of {}'.format(i + 1, len(dataloader)))
        for j in range(adv_img.shape[0]):
            cur_img = adv_img[j].detach()

            if target == 'HighResolution':
                inv_norm = cd.NormalizeInverse(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

                saved_adv = inv_norm(adv_img[j].detach().clone())
                saved_adv = torch.clamp(saved_adv, 0, 1)

                true_idx = int(true_label_cpu[j])
                pred_idx = int(pred_label_cpu[j])

                save_image(
                    saved_adv,
                    './results/examples/{}/{}/example_{}_{}_true_{}_pred_{}.png'.format(
                        target, mode, i, j, true_idx, pred_idx
                    )
                )
            else:
                true_idx = int(true_label_cpu[j])
                pred_idx = int(pred_label_cpu[j])

                save_image(
                    cur_img,
                    './results/examples/{}/{}/example_{}_{}_true_{}_pred_{}.png'.format(
                        target, mode, i, j, true_idx, pred_idx
                    )
                )

    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    img_np = np.concatenate(img_np, axis=0)
    adv_img_np = np.concatenate(adv_img_np, axis=0)

    np.save('./npy/{}/{}_true_labels'.format(target, mode), true_labels)
    np.save('./npy/{}/{}_pred_labels'.format(target, mode), pred_labels)
    np.save('./npy/{}/{}_img_np'.format(target, mode), img_np)
    np.save('./npy/{}/{}_adv_img_np'.format(target, mode), adv_img_np)

    print(target)
    print('Correctly Classified in full dataset under attack: {}'.format(n_correct.item()))
    print('Accuracy under attacks in {} full dataset: {}%\n'.format(
        target, 100 * n_correct.item() / dataset_size)
    )

    return true_labels, pred_labels, n_correct.item()


print('\nLOADING CONFIGURATIONS...')
TARGET, LR_TARGET_MODEL, EPOCHS_TARGET_MODEL, L_INF_BOUND, EPOCHS, LR, ALPHA, BETA, GAMMA, KAPPA, C, N_STEPS_D, N_STEPS_G, IS_RELATIVISTIC = load_hyperparameters('hyperparams.json')

print('\nCREATING NECESSARY DIRECTORIES...')
create_dirs()

print('\nCHECKING FOR CUDA...')
use_cuda = True
print('CUDA Available: ', torch.cuda.is_available())
device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')

print('\nPREPARING DATASETS...')
(
    dataloader,
    target_model,
    batch_size,
    l_inf_bound,
    n_labels,
    n_channels,
    dataset_size
) = init_params(TARGET)

if TARGET != 'HighResolution':
    print('CHECKING FOR PRETRAINED TARGET MODEL...')
    try:
        pretrained_target = './checkpoints/target/{}_bs_{}_lbound_{}.pth'.format(TARGET, batch_size, l_inf_bound)
        target_model.load_state_dict(torch.load(pretrained_target))
        target_model.eval()
    except FileNotFoundError:
        print('\tNO PRETRAINED MODEL FOUND... TRAINING TARGET FROM SCRATCH...')
        train_target_model(
            target=TARGET,
            target_model=target_model,
            epochs=EPOCHS_TARGET_MODEL,
            dataloader=dataloader,
            dataset_size=dataset_size
        )

print('TARGET LOADED!')

print('\nTESTING CLEAN PERFORMANCE ON FULL DATASET...')
clean_true_labels, clean_pred_labels, clean_correct = test_clean_performance(
    target=TARGET,
    dataloader=dataloader,
    target_model=target_model,
    dataset_size=dataset_size,
    mode='train'
)

print('\nTRAINING ADVGAN...')
advGAN = AdvGAN_Attack(
    device,
    target_model,
    n_labels,
    n_channels,
    target=TARGET,
    lr=LR,
    l_inf_bound=l_inf_bound,
    alpha=ALPHA,
    beta=BETA,
    gamma=GAMMA,
    kappa=KAPPA,
    c=C,
    n_steps_D=N_STEPS_D,
    n_steps_G=N_STEPS_G,
    is_relativistic=IS_RELATIVISTIC
)
advGAN.train(dataloader, EPOCHS)

print('\nLOADING TRAINED ADVGAN!')
adv_GAN_path = './checkpoints/AdvGAN/G_epoch_{}.pth'.format(EPOCHS)
adv_GAN = models.Generator(n_channels, n_channels, TARGET).to(device)
adv_GAN.load_state_dict(torch.load(adv_GAN_path))
adv_GAN.eval()

print('\nTESTING PERFORMANCE OF ADVGAN ON FULL DATASET...')
adv_true_labels, adv_pred_labels, adv_correct = test_attack_performance(
    target=TARGET,
    dataloader=dataloader,
    mode='train',
    adv_GAN=adv_GAN,
    target_model=target_model,
    batch_size=batch_size,
    l_inf_bound=l_inf_bound,
    dataset_size=dataset_size
)

print('\nSUMMARY:')
print('Full Dataset Clean Accuracy: {:.2f}%'.format(100 * clean_correct / dataset_size))
print('Full Dataset Adv Accuracy: {:.2f}%'.format(100 * adv_correct / dataset_size))
print('Accuracy Drop: {:.2f}%'.format(100 * (clean_correct - adv_correct) / dataset_size))