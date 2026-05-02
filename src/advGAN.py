# Adversarial Attacks with AdvGAN and AdvRaGAN
# Modified from https://github.com/mathcbc/advGAN_pytorch/blob/master/advGAN.py

import matplotlib
matplotlib.use("Agg")

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import models


def init_weights(m):
    """Custom weights initialization called on G and D."""
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(
        self,
        device,
        model,
        n_labels,
        n_channels,
        target,
        lr,
        l_inf_bound,
        alpha,
        beta,
        gamma,
        kappa,
        c,
        n_steps_D,
        n_steps_G,
        is_relativistic,
        checkpoint_dir,
        loss_dir
    ):
        self.device = device
        self.n_labels = n_labels
        self.model = model
        self.target = target
        self.lr = lr
        self.l_inf_bound = l_inf_bound
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kappa = kappa
        self.c = c
        self.n_steps_D = n_steps_D
        self.n_steps_G = n_steps_G
        self.is_relativistic = is_relativistic

        self.checkpoint_dir = checkpoint_dir
        self.loss_dir = loss_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)

        self.G = models.Generator(n_channels, n_channels, target).to(device)
        self.D = models.Discriminator(n_channels).to(device)

        self.G.apply(init_weights)
        self.D.apply(init_weights)

        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr)

        self.model.eval()

    def _build_adv_images(self, x):
        perturbation = self.G(x)

        if perturbation.shape[-2:] != x.shape[-2:]:
            perturbation = F.interpolate(
                perturbation,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        adv_images = torch.clamp(perturbation, -self.l_inf_bound, self.l_inf_bound) + x
        adv_images = torch.clamp(adv_images, 0, 1)

        return perturbation, adv_images

    def train_batch(self, x, labels):
        last_loss_D = 0.0

        # Optimize D
        for _ in range(self.n_steps_D):
            _, adv_images = self._build_adv_images(x)

            self.D.zero_grad()

            logits_real, pred_real = self.D(x)
            logits_fake, pred_fake = self.D(adv_images.detach())

            real = torch.ones_like(pred_real, device=self.device)

            if self.is_relativistic:
                loss_D_real = torch.mean((logits_real - torch.mean(logits_fake) - real) ** 2)
                loss_D_fake = torch.mean((logits_fake - torch.mean(logits_real) + real) ** 2)
                loss_D = (loss_D_fake + loss_D_real) / 2
            else:
                loss_D_real = F.mse_loss(
                    pred_real,
                    torch.ones_like(pred_real, device=self.device)
                )
                loss_D_fake = F.mse_loss(
                    pred_fake,
                    torch.zeros_like(pred_fake, device=self.device)
                )
                loss_D = loss_D_fake + loss_D_real

            loss_D.backward()
            self.optimizer_D.step()
            last_loss_D = loss_D.item()

        last_loss_G = 0.0
        last_loss_G_gan = 0.0
        last_loss_hinge = 0.0
        last_loss_adv = 0.0

        # Optimize G
        for _ in range(self.n_steps_G):
            self.G.zero_grad()

            perturbation, adv_images = self._build_adv_images(x)

            perturbation_norm = torch.mean(
                torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
            )
            loss_hinge = torch.max(
                torch.zeros(1, device=self.device),
                perturbation_norm - self.c
            )

            logits_model = self.model(adv_images)
            probs_model = F.softmax(logits_model, dim=1)

            onehot_labels = torch.eye(self.n_labels, device=self.device)[labels]

            real_class_prob = torch.sum(onehot_labels * probs_model, dim=1)
            target_class_prob, _ = torch.max(
                (1 - onehot_labels) * probs_model - onehot_labels * 10000,
                dim=1
            )

            loss_adv = torch.max(
                real_class_prob - target_class_prob,
                self.kappa * torch.ones_like(target_class_prob)
            )
            loss_adv = torch.sum(loss_adv)

            logits_real, pred_real = self.D(x)
            logits_fake, pred_fake = self.D(adv_images)

            real = torch.ones_like(pred_real, device=self.device)

            if self.is_relativistic:
                loss_G_real = torch.mean((logits_real - torch.mean(logits_fake) + real) ** 2)
                loss_G_fake = torch.mean((logits_fake - torch.mean(logits_real) - real) ** 2)
                loss_G_gan = (loss_G_real + loss_G_fake) / 2
            else:
                loss_G_gan = F.mse_loss(
                    pred_fake,
                    torch.ones_like(pred_fake, device=self.device)
                )

            loss_G = self.gamma * loss_adv + self.alpha * loss_G_gan + self.beta * loss_hinge

            loss_G.backward()
            self.optimizer_G.step()

            last_loss_G = loss_G.item()
            last_loss_G_gan = loss_G_gan.item()
            last_loss_hinge = loss_hinge.item()
            last_loss_adv = loss_adv.item()

        return (
            last_loss_D,
            last_loss_G,
            last_loss_G_gan,
            last_loss_hinge,
            last_loss_adv
        )

    def _save_loss_plot(self, values, filename):
        plt.figure()
        plt.plot(values)
        plt.savefig(os.path.join(self.loss_dir, filename))
        plt.close()

    def train(self, train_dataloader, epochs):
        loss_D = []
        loss_G = []
        loss_G_gan = []
        loss_hinge = []
        loss_adv = []

        for epoch in range(1, epochs + 1):
            loss_D_sum = 0.0
            loss_G_sum = 0.0
            loss_G_gan_sum = 0.0
            loss_hinge_sum = 0.0
            loss_adv_sum = 0.0

            for _, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                (
                    loss_D_batch,
                    loss_G_batch,
                    loss_G_gan_batch,
                    loss_hinge_batch,
                    loss_adv_batch
                ) = self.train_batch(images, labels)

                loss_D_sum += loss_D_batch
                loss_G_sum += loss_G_batch
                loss_adv_sum += loss_adv_batch
                loss_G_gan_sum += loss_G_gan_batch
                loss_hinge_sum += loss_hinge_batch

            batch_count = len(train_dataloader)

            epoch_loss_D = loss_D_sum / batch_count
            epoch_loss_G = loss_G_sum / batch_count
            epoch_loss_adv = loss_adv_sum / batch_count
            epoch_loss_G_gan = loss_G_gan_sum / batch_count
            epoch_loss_hinge = loss_hinge_sum / batch_count

            print(
                "Epoch {}:\n"
                "Loss D: {},\n"
                "Loss G: {},\n"
                "\t-Loss Adv: {},\n"
                "\t-Loss G GAN: {},\n"
                "\t-Loss Hinge: {}\n".format(
                    epoch,
                    epoch_loss_D,
                    epoch_loss_G,
                    epoch_loss_adv,
                    epoch_loss_G_gan,
                    epoch_loss_hinge
                )
            )

            loss_D.append(epoch_loss_D)
            loss_G.append(epoch_loss_G)
            loss_adv.append(epoch_loss_adv)
            loss_G_gan.append(epoch_loss_G_gan)
            loss_hinge.append(epoch_loss_hinge)

            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"G_epoch_{epoch}.pth"
            )
            torch.save(self.G.state_dict(), checkpoint_path)

            self._save_loss_plot(loss_D, "loss_D.png")
            self._save_loss_plot(loss_G, "loss_G.png")
            self._save_loss_plot(loss_adv, "loss_adv.png")
            self._save_loss_plot(loss_G_gan, "loss_G_gan.png")
            self._save_loss_plot(loss_hinge, "loss_hinge.png")

        return {
            "loss_D": loss_D,
            "loss_G": loss_G,
            "loss_adv": loss_adv,
            "loss_G_gan": loss_G_gan,
            "loss_hinge": loss_hinge
        }