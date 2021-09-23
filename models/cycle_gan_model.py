import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import LightCNN as ID_pre
from torchvision import transforms
from PIL import Image
import torch.nn as nn
# from loss import Huber
# import sys
# sys.path.append(r"/hdd01/wanghuijiao/CG02/models")

# ref: https://github.com/kingsj0405/ciplab-NTIRE-2020/blob/17a8f3d5e0531d5ce0a0a4af07ab048cd94121cb/utils.py
import torch

def Huber(input, target, delta=0.01, reduce=True):
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)

    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * torch.pow(quadratic, 2) + delta * linear
    
    if reduce:
        return torch.mean(losses)
    else:
        return losses


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(
            no_dropout=True)  # default CycleGAN did not use dropout

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.opt.loss_flag == 'cyclegan':
            self.loss_names = ['D_A', 'G_A', 'cycle_A',
                               'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        elif self.opt.loss_flag == "pixel":
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B',
                               'G_B', 'cycle_B', 'idt_B',  'pixel_A', 'pixel_B']
        elif self.opt.loss_flag == 'pixel_huber':
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 
                                'D_B', 'G_B', 'cycle_B', 'idt_B',  'pixel_A', 'pixel_B']
        elif self.opt.loss_flag == "fea":
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B',
                               'G_B', 'cycle_B', 'idt_B',  'fea_dis_A', 'fea_dis_B']
        elif self.opt.loss_flag == "pix_fea":
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B',
                               'cycle_B', 'idt_B',  'pixel_A', 'pixel_B', 'fea_dis_A', 'fea_dis_B']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        # combine visualizations for A and B
        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)
            # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            # define GAN loss.
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(
            ), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(
            ), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def resize_output_of_netG(self, output_of_netG):
        # the output tensor of netG is [1, 3, 256, 256], we are going to resize it to [1, 3, 128, 128]
        # transfarm tensor [3, 256, 256] to PIL image [256, 256, 3]
        PILimage = transforms.ToPILImage()(output_of_netG[0, :, :, :].cpu())
        # [256, 256, 3] -> [128, 128, 3]
        resize_PILimage = PILimage.resize((128, 128))
        # [128, 128, 3] -> [1, 3, 128, 128]
        resized_output_of_netG = transforms.ToTensor()(resize_PILimage).unsqueeze(dim=0)
        # print("resize_PILimage size: ", resize_PILimage, "resized_output_of_netG size: ", resized_output_of_netG.size())

        return resized_output_of_netG

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        loss_flag = self.opt.loss_flag
        lambda_C = self.opt.lambda_fea
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(
                self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(
                self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(
            self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(
            self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients

        if loss_flag == "cyclegan":
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                self.loss_idt_A + self.loss_idt_B
        elif loss_flag == "pixel":
            # pixel consistency loss
            self.loss_pixel_A = self.criterionCycle(self.fake_B, self.real_B) * lambda_A
            self.loss_pixel_B = self.criterionCycle(self.fake_A, self.real_A) * lambda_B
            # Combined loss
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                self.loss_idt_A + self.loss_idt_B + self.loss_pixel_A + self.loss_pixel_B
        elif loss_flag == "pixel_huber":
            # pixel consistency loss
            self.loss_pixel_A = Huber(self.fake_B, self.real_B) * lambda_A
            self.loss_pixel_B = Huber(self.fake_A, self.real_A) * lambda_B
            # Combined loss
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                self.loss_idt_A + self.loss_idt_B + self.loss_pixel_A + self.loss_pixel_B
        elif loss_flag == "fea":
            # define LightCNN29 and cosine loss
            LightCNN29 = ID_pre.define_R(gpu_ids=[
                                         0], lightcnn_path='/ssd01/wanghuijiao/CG/LightCNN29/LightCNN_29Layers_V2_checkpoint.pth').to(torch.device('cuda'))
            Cos_loss = nn.CosineEmbeddingLoss().to(torch.device('cuda'))
            # Feature consistency loss
            self.real_A_fea = LightCNN29(
                self.resize_output_of_netG(self.real_A))
            self.fake_A_fea = LightCNN29(
                self.resize_output_of_netG(self.fake_A))
            self.real_B_fea = LightCNN29(
                self.resize_output_of_netG(self.real_B))
            self.fake_B_fea = LightCNN29(
                self.resize_output_of_netG(self.fake_B))
            self.loss_fea_dis_A = Cos_loss(self.fake_A_fea, self.real_A_fea, torch.ones(
                (self.real_A_fea.shape[0], 1)).cuda()) * lambda_C
            self.loss_fea_dis_B = Cos_loss(self.fake_B_fea, self.real_B_fea, torch.ones(
                (self.real_A_fea.shape[0], 1)).cuda()) * lambda_C

            # Combined loss
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                self.loss_idt_A + self.loss_idt_B - self.loss_fea_dis_A - self.loss_fea_dis_B
            # print("################## fea_dis_A: ", self.fea_dis_A, "\n ################## self.loss_G_A: ", self.loss_G_A)
        elif loss_flag == "pix_fea":
            # pixel consistency loss
            self.loss_pixel_A = self.criterionCycle(
                self.fake_B, self.real_B) * lambda_A
            self.loss_pixel_B = self.criterionCycle(
                self.fake_A, self.real_A) * lambda_B
            # define LightCNN29 and cosine loss
            LightCNN29 = ID_pre.define_R(gpu_ids=[
                                         0], lightcnn_path='/ssd01/wanghuijiao/CG/LightCNN29/LightCNN_29Layers_V2_checkpoint.pth').to(torch.device('cuda'))
            Cos_loss = nn.CosineEmbeddingLoss().to(torch.device('cuda'))
            # Feature consistency loss
            self.real_A_fea = LightCNN29(
                self.resize_output_of_netG(self.real_A))
            self.fake_A_fea = LightCNN29(
                self.resize_output_of_netG(self.fake_A))
            self.real_B_fea = LightCNN29(
                self.resize_output_of_netG(self.real_B))
            self.fake_B_fea = LightCNN29(
                self.resize_output_of_netG(self.fake_B))
            self.loss_fea_dis_A = Cos_loss(self.fake_A_fea, self.real_A_fea, torch.ones(
                (self.real_A_fea.shape[0], 1)).cuda()) * lambda_C
            self.loss_fea_dis_B = Cos_loss(self.fake_B_fea, self.real_B_fea, torch.ones(
                (self.real_A_fea.shape[0], 1)).cuda()) * lambda_C
            # Combined loss
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + \
                self.loss_idt_B + self.loss_pixel_A + self.loss_pixel_B - \
                self.loss_fea_dis_A - self.loss_fea_dis_B

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
