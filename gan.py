import logging

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.ResNet8 import ResNet8
from models.ResNet18 import ResNet18
from models.CNNMnist import CNNMnist

logger = logging.getLogger("logger")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator_MNIST(nn.Module):
    def __init__(self):
        super(Generator_MNIST, self).__init__()

        self.init_size = 28 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Generator_CIFAR(nn.Module):
    def __init__(self, the_dim=3):
        super(Generator_CIFAR, self).__init__()

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, the_dim, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


CGAN_E1_DEFAULT = 10
CGAN_E2_DEFAULT = 5


def resolve_cgan_rounds(params):
    """Return the optimization rounds for the two FilterFL synthesis stages."""

    rounds = []
    for key, default in (("cgan_e1", CGAN_E1_DEFAULT), ("cgan_e2", CGAN_E2_DEFAULT)):
        raw = params.get(key, default)
        if type(raw) is bool or not isinstance(raw, int):
            raise ValueError(f"{key} must be an integer; received {raw!r}")
        if raw < 1:
            raise ValueError(f"{key} must be at least 1; received {raw!r}")
        rounds.append(int(raw))
    return tuple(rounds)


def gan_defense(d1, d2, params, the_epoch):
    lr = 0.0005
    b1 = 0.5
    b2 = 0.999
    latent_dim = 100

    def _canonical_type(t: str) -> str:
        t_low = str(t).lower()
        if t_low in ["mnist"]:
            return "MNIST"
        if t_low in ["cifar10", "cifar-10"]:
            return "CIFAR10"
        if t_low in ["cifar100", "cifar-100"]:
            return "CIFAR100"
        if t_low in ["gtsrb"]:
            return "GTSRB"
        if t_low in ["tinyimagenet", "tiny-imagenet"]:
            return "TinyImagenet"
        return str(t)

    ds_type = _canonical_type(params.get("type", "CIFAR10"))
    if ds_type == "MNIST":
        classnum = 10
        dims = 1
    elif ds_type == "CIFAR100":
        classnum = 100
        dims = 3
    elif ds_type == "GTSRB":
        classnum = 43
        dims = 3
    elif ds_type == "TinyImagenet":
        classnum = 200
        dims = 3
    else:
        classnum = 10
        dims = 3
    try:
        image_num = int(params.get("trigger_batch_size", params.get("trigger_image_num", 32)))
    except Exception:
        image_num = 32
    if image_num <= 0:
        image_num = 32
    logger.info(f"[trigger_config] image_num={image_num}")
    if ds_type == "MNIST":
        discriminator1 = CNNMnist(num_classes=classnum)
        discriminator2 = CNNMnist(num_classes=classnum)
        generator = Generator_MNIST()
    elif ds_type == "CIFAR100":
        discriminator1 = ResNet18(num_classes=classnum, in_channels=3)
        discriminator2 = ResNet18(num_classes=classnum, in_channels=3)
        generator = Generator_CIFAR(the_dim=dims)
    elif ds_type == "GTSRB":
        discriminator1 = ResNet8(num_classes=classnum)
        discriminator2 = ResNet8(num_classes=classnum)
        generator = Generator_CIFAR(the_dim=dims)
    elif ds_type == "TinyImagenet":
        discriminator1 = ResNet18(num_classes=classnum, in_channels=3)
        discriminator2 = ResNet18(num_classes=classnum, in_channels=3)
        generator = Generator_CIFAR(the_dim=dims)
    else:
        discriminator1 = ResNet8(num_classes=classnum)
        discriminator2 = ResNet8(num_classes=classnum)
        generator = Generator_CIFAR(the_dim=dims)

    n_epochs1, n_epochs2 = resolve_cgan_rounds(params)
    logger.info(f"[trigger_config] cgan_e1={n_epochs1} cgan_e2={n_epochs2}")

    try:
        d1_dev = {
            k: (v.detach().to(device) if torch.is_tensor(v) else torch.as_tensor(v).to(device))
            for k, v in d1.items()
        }
        d2_dev = {
            k: (v.detach().to(device) if torch.is_tensor(v) else torch.as_tensor(v).to(device))
            for k, v in d2.items()
        }
    except Exception:
        d1_dev, d2_dev = d1, d2
    discriminator1.load_state_dict(d1_dev, strict=False)
    discriminator2.load_state_dict(d2_dev, strict=False)

    generator.to(device)
    discriminator1.to(device)
    discriminator2.to(device)
    try:
        for m in (generator, discriminator1, discriminator2):
            for p in m.parameters():
                try:
                    p.data = p.data.to(device)
                except Exception:
                    pass
            for name, buf in m.named_buffers():
                try:
                    setattr(m, name, buf.to(device))
                except Exception:
                    try:
                        m._buffers[name] = m._buffers[name].to(device)
                    except Exception:
                        pass
    except Exception:
        pass

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    ori_imgs = []

    import math

    for c in range(classnum):
        para = None
        loss_min = float("inf")
        Alpha = 0.75
        generator.apply(weights_init_normal)
        for epoch in range(n_epochs1):
            optimizer_G.zero_grad()
            z = torch.randn((image_num, latent_dim), device=device)
            gen_imgs = generator(z)
            out1 = torch.softmax(discriminator1(gen_imgs), dim=1)
            out1 = torch.std(out1, dim=1)
            out1 = torch.mean(out1)

            out2 = torch.softmax(discriminator2(gen_imgs), dim=1)
            out2 = out2[:, c]
            out2 = torch.mean(out2)
            g_loss = Alpha * out1 + (1 - Alpha) * (1 - out2)
            g_loss.backward()
            optimizer_G.step()

            try:
                g_loss_val = float(g_loss.detach().item())
            except Exception:
                g_loss_val = float("inf")
            if math.isfinite(g_loss_val) and g_loss_val < loss_min:
                loss_min = g_loss_val
                para = generator.state_dict()

        if para is None:
            para = generator.state_dict()

        generator.load_state_dict(para)
        generator.to(device)
        z = torch.randn((image_num, latent_dim), device=device)
        imgs = generator(z)
        ori_imgs.append(imgs)

    # Broadcast the synthesized single-channel pattern across input channels.
    tri_imgs = []
    if ds_type == "MNIST":
        generator = Generator_MNIST()
    else:
        generator = Generator_CIFAR(the_dim=1)
    generator = generator.to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))

    para = None
    import math

    loss_min = float("inf")
    generator.apply(weights_init_normal)
    generator.to(device)
    this_imgs = ori_imgs[0]
    if classnum > 1:
        other_imgs = torch.cat(ori_imgs[1:])
    else:
        other_imgs = this_imgs.detach()
    for epoch in range(n_epochs2):
        optimizer_G.zero_grad()
        z = torch.randn((image_num, latent_dim), device=device)
        gen_trigger = generator(z)
        g_loss = 0
        new_gen_img = other_imgs.detach() + gen_trigger.repeat([classnum - 1, dims, 1, 1])
        out1 = torch.softmax(discriminator1(new_gen_img), dim=1)
        out1 = torch.std(out1, dim=1)
        out1 = torch.mean(out1, dim=0)

        out2 = torch.softmax(discriminator2(new_gen_img), dim=1)
        out2 = torch.mean(out2, dim=0)
        out2 = out2[0]

        g_loss = g_loss + 0.75 * out1 + (1 - 0.75) * (1 - out2)

        new_gen_trigger = gen_trigger.repeat([1, dims, 1, 1])
        new_gen_img = this_imgs.detach() - new_gen_trigger

        out1 = torch.softmax(discriminator1(new_gen_img), dim=1)
        out1 = torch.std(out1, dim=1)
        out1 = torch.mean(out1, dim=0)

        out2 = torch.softmax(discriminator2(new_gen_img), dim=1)
        out2 = torch.mean(out2, dim=0)
        out2 = out2[0]

        out3 = torch.softmax(discriminator1(new_gen_trigger), dim=1)
        out3 = torch.mean(out3, dim=0)
        out3 = out3[0]

        g_loss = g_loss + 0.75 * out1 + (1 - 0.75) * out2 + (1 - 0.75) * out3
        g_loss.backward()
        optimizer_G.step()
        try:
            g_loss_val = float(g_loss.detach().item())
        except Exception:
            g_loss_val = float("inf")
        if math.isfinite(g_loss_val) and g_loss_val < loss_min:
            loss_min = g_loss_val
            para = generator.state_dict()

    if para is None:
        para = generator.state_dict()
    generator.load_state_dict(para)
    generator.to(device)
    z = torch.randn((image_num, latent_dim), device=device)
    with torch.no_grad():
        imgs = generator(z)
        tri_imgs.append(imgs.repeat([1, dims, 1, 1]))
    return tri_imgs
