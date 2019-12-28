from os.path import join as PJ
import torch
import torch.nn as nn

from networks.resnet_based import ResNet101EncoderBridge, ResNetClassifier, FCN8sSegment
from utils import get_model_list, get_scheduler

import torchvision.utils as vutils

"""
Source(nn.Module)
    Source only (Single-task and Multi-task).
"""


################################################################################
# Source only
################################################################################
class SourceOnly(nn.Module):
    """ Source only (Single-task and Multi-task)0. """
    def __init__(self, config):
        super().__init__()
        self._initial_networks(config)
        self._initial_optimizer(config)
        self.num_class = config['num_class']

        # For semantic segmentation
        self.id_to_color = torch.tensor(
            [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
             [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
             [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
             [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=torch.float32)

    def _initial_networks(self, config):
        init = config['init']

        # Create model
        self.encoder = ResNet101EncoderShare()

        # Create classification model
        if config['classifier'] == 'ResNet101':
            self.enc_bridge1 = ResNet101EncoderBridge()
            self.classifier = ResNetClassifier(config['num_class']['cls'], init)

        # Create semantic segmentation model
        if config['segmentor'] == 'ResNet101':
            self.enc_bridge2 = ResNet101EncoderBridge()
            self.segmentor = FCN8sSegment(config['num_class']['seg'], init)

        # Turn off batch normalize
        if 'enc' in config['FBN']:
            self._freeze_batch_norm(self.encoder)
        if 'bdg1' in config['FBN']:
            self._freeze_batch_norm(self.enc_bridge1)
        if 'bdg2' in config['FBN']:
            self._freeze_batch_norm(self.enc_bridge2)

        if config['pretrained_checkpoint']:
            self._pretrained(config)
            self._freeze(config['freeze_list'])

    def _freeze_batch_norm(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.affine = False
                m.track_running_stats = False
                # In MTL
                # m.affine = True
                # m.track_running_stats = True

    def _freeze(self, freeze_list):
        freeze_param = []
        print("Freeze:", end=" ")
        # Add model into freeze_param list
        if 'encoder' in freeze_list:
            freeze_param += list(self.encoder.parameters())
            print("[encoder]", end=" | ")
        if 'classifier' in freeze_list:
            freeze_param += list(self.classifier.parameters())
            print("[classifier]", end=" | ")
        if 'segmentor' in freeze_list:
            freeze_param += list(self.segmentor.parameters())
            print("[segmentor]", end=" | ")
        if 'enc_bridge1' in freeze_list:
            freeze_param += list(self.enc_bridge1.parameters())
            print("[enc_bridge1]", end=" | ")
        if 'enc_bridge2' in freeze_list:
            freeze_param += list(self.enc_bridge2.parameters())
            print("[enc_bridge2]", end=" | ")
        print("\n")

        # Freeze parameters
        for param in freeze_param:
            param.requires_grad = False

    def _initial_optimizer(self, config):

        gen_params = list(self.encoder.parameters())
        if config['classifier']:
            gen_params += list(self.enc_bridge1.parameters())
            gen_params += list(self.classifier.parameters())
        if config['segmentor']:
            gen_params += list(self.enc_bridge2.parameters())
            gen_params += list(self.segmentor.parameters())

        self.optimizer = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                           lr=config['lr'], betas=(0.5, 0.999), weight_decay=config['weight_decay'])
        self.scheduler = get_scheduler(self.optimizer, config)

    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    ##################################################
    # Test Forward (Basic)
    ##################################################
    def classify(self, x, domain=None):
        """ Classification forward """
        self.eval()
        h = self.encoder(x, trans=False, task='cls')
        h = self.enc_bridge1(h, task='cls')
        predicts = self.classifier(h).type(torch.float)
        return predicts

        # predicts, features = self.classifier(h)
        # predicts = predicts.type(torch.float)
        # self.train()
        # return predicts, features

    def segment(self, x, domain=None):
        """ Segmetation forward """
        self.eval()
        h = self.encoder(x, trans=False, task='seg')
        pool3, pool4, pool5 = self.enc_bridge2(h, task='seg')
        predicts = self.segmentor(pool3, pool4, pool5, x.shape).type(torch.float)
        predicts = torch.argmax(predicts, dim=1, keepdim=True)
        return predicts

    ##################################################
    # Calculate loss (Basic)
    ##################################################
    def compute_classify_loss(self, h, targets):
        targets = targets.view(-1)
        predicts = self.classifier(h).type(torch.float)
        criterion = nn.CrossEntropyLoss()
        return criterion(predicts, targets)

    def compute_segmentation_loss(self, pool3, pool4, pool5, x_shape, targets):
        predicts = self.segmentor(pool3, pool4, pool5, x_shape).type(torch.float)
        predicts = nn.LogSoftmax(dim=1)(predicts)
        criterion = torch.nn.NLLLoss(ignore_index=19)
        loss = criterion(predicts, targets)
        return loss

    ##################################################
    # Training (Basic)
    ##################################################
    def model_update(self, x, targets, config):

        # Initialization
        self.loss_sup_total = 0
        self.optimizer.zero_grad()

        if config['classifier']:
            h = self.encoder(x[0], trans=False, task='cls')
            h = self.enc_bridge1(h, task='cls')
            self.loss_cls = self.compute_classify_loss(h, targets[0])

        if config['segmentor']:
            h = self.encoder(x[-1], trans=False, task='seg')
            pool3, pool4, pool5 = self.enc_bridge2(h, task='seg')
            self.loss_seg = self.compute_segmentation_loss(pool3, pool4, pool5, x.shape, targets[-1])

        self.loss_sup_total = config['cls_w'] * self.loss_cls +\
                              config['seg_w'] * self.loss_seg

        self.loss_sup_total.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    ##################################################
    # Saving and loading model (Basic)
    ##################################################
    def save(self, config, snapshot_dir, iterations):
        """ Save encoder, classifier and optimizers """
        enc_name = PJ(snapshot_dir, f'enc_{iterations:08d}.pt')
        bdg1_name = PJ(snapshot_dir, f'bdg1_{iterations:08d}.pt')
        bdg2_name = PJ(snapshot_dir, f'bdg2_{iterations:08d}.pt')
        cls_name = PJ(snapshot_dir, f'cls_{iterations:08d}.pt')
        seg_name = PJ(snapshot_dir, f'seg_{iterations:08d}.pt')
        opt_name = PJ(snapshot_dir, f'optimizer.pt')

        torch.save(self.encoder.state_dict(), enc_name)
        torch.save(self.optimizer.state_dict(), opt_name)

        if config['classifier']:
            torch.save(self.enc_bridge1.state_dict(), bdg1_name)
            torch.save(self.classifier.state_dict(), cls_name)
        if config['segmentor']:
            torch.save(self.enc_bridge2.state_dict(), bdg2_name)
            torch.save(self.segmentor.state_dict(), seg_name)
        print(f"Saving models in {snapshot_dir} finished.")

    def resume(self, config, checkpoint_dir):
        """ Resume training progress. """
        # Load encoder
        last_model_name = get_model_list(checkpoint_dir, "enc")
        state_dict = torch.load(last_model_name)
        self.encoder.load_state_dict(state_dict)
        iterations = int(last_model_name[-11:-3])

        # Load classifier
        if config['classifier']:
            last_model_name = get_model_list(checkpoint_dir, "cls")
            state_dict = torch.load(last_model_name)
            self.classifier.load_state_dict(state_dict)
            last_model_name = get_model_list(checkpoint_dir, "bdg1")
            state_dict = torch.load(last_model_name)
            self.enc_bridge1.load_state_dict(state_dict)
        # Load Segmentor
        if config['segmentor']:
            last_model_name = get_model_list(checkpoint_dir, "seg")
            state_dict = torch.load(last_model_name)
            self.segmentor.load_state_dict(state_dict)
            last_model_name = get_model_list(checkpoint_dir, "bdg2")
            state_dict = torch.load(last_model_name)
            self.enc_bridge2.load_state_dict(state_dict)

        # Load optimizers
        state_dict = torch.load(PJ(checkpoint_dir, 'optimizer.pt'))
        self.optimizer.load_state_dict(state_dict)
        # Reinitilize schedulers
        self.scheduler = get_scheduler(self.optimizer, config, iterations)
        print(f'Resume from iteration {iterations}')
        return iterations

    def _pretrained(self, config):
        """ Load pretrained model. """
        print("Load pretrained model:")
        if config['enc_weights']:
            enc_name = PJ(config['pretrained_checkpoint'], config['enc_weights'])
            state_dict = torch.load(enc_name)
            self.encoder.load_state_dict(state_dict)
            print(f"{config['enc_weights']} ", end="  ")

        if config['bdg1_weights']:
            bdg_name = PJ(config['pretrained_checkpoint'], config['bdg1_weights'])
            state_dict = torch.load(bdg_name)
            self.enc_bridge1.load_state_dict(state_dict)
            print(f"{config['bdg1_weights']} ", end="  ")

        if config['bdg2_weights']:
            bdg_name = PJ(config['pretrained_checkpoint'], config['bdg2_weights'])
            state_dict = torch.load(bdg_name)
            self.enc_bridge2.load_state_dict(state_dict)
            print(f"{config['bdg2_weights']} ", end="  ")

        if config['cls_weights']:
            cls_name = PJ(config['pretrained_checkpoint'], config['cls_weights'])
            state_dict = torch.load(cls_name)
            self.classifier.load_state_dict(state_dict)
            print(f"{config['cls_weights']} ", end="  ")

        if config['seg_weights']:
            seg_name = PJ(config['pretrained_checkpoint'], config['seg_weights'])
            state_dict = torch.load(seg_name)
            self.segmentor.load_state_dict(state_dict)
            print(f"{config['seg_weights']} ", end="  ")
        print(f"\nLoad pretrained model from {config['pretrained_checkpoint']}\n")

    def load_weights(self, checkpoint_dir, enc_weight=None, cls_weight=None, seg_weight=None, bdg1_weight=None, bdg2_weight=None):
        """ Load trained model. """
        print(f"Loading model weights:")
        # Load encoder
        if enc_weight:
            state_dict = torch.load(PJ(checkpoint_dir, enc_weight))
            self.encoder.load_state_dict(state_dict)
            print(f"{enc_weight}", end="  ")
        # Load classifier
        if cls_weight:
            state_dict = torch.load(PJ(checkpoint_dir, cls_weight))
            self.classifier.load_state_dict(state_dict)
            print(f"{cls_weight}", end="  ")
        # Load segmentor
        if seg_weight:
            state_dict = torch.load(PJ(checkpoint_dir, seg_weight))
            self.segmentor.load_state_dict(state_dict)
            print(f"{seg_weight}", end="  ")
        # Load encoder bridge (cls)
        if bdg1_weight:
            state_dict = torch.load(PJ(checkpoint_dir, bdg1_weight))
            self.enc_bridge1.load_state_dict(state_dict)
            print(f"{bdg1_weight}", end="  ")
        # Load encoder bridge (seg)
        if bdg2_weight:
            state_dict = torch.load(PJ(checkpoint_dir, bdg2_weight))
            self.enc_bridge2.load_state_dict(state_dict)
            print(f"{bdg2_weight}", end="  ")
        print("finished.\n")

    #################################################################
    # Record images (Basic)
    #################################################################
    def record_segment_images(self, labels, images, segment_directory, name, results=None, domain=None, _scale=0.5):
        # Skip background
        num_class = self.num_class['seg'] - 1
        num_image, h, w = labels.shape

        labels = labels.cpu().unsqueeze(1)
        results = self.segment(images, domain).cpu() if results is None else results
        labels_visual = torch.zeros((num_image, 3, h, w), dtype=torch.float32)
        results_visual = torch.zeros((num_image, 3, h, w), dtype=torch.float32)

        # Replace background pixel in results by 19
        mask_background = (labels == 19)
        results[mask_background] = 19.

        # Draw color (Categories)
        for i in range(num_class):
            color = self.id_to_color[i].reshape(1, 3, 1, 1)
            # Label (using mask which size is equal to _visual)
            l_mask = (labels == i).repeat(1, 3, 1, 1).type(torch.float32)
            labels_visual += l_mask * color
            # Result (using mask which size is equal to _visual)
            r_mask = (results == i).repeat(1, 3, 1, 1).type(torch.float32)
            results_visual += r_mask * color

        # Denormalize
        mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        images = images.cpu() * std + mean
        labels_visual = labels_visual / 255.
        results_visual = results_visual / 255.
        # Save original images, labels, results
        visual = torch.cat([images, labels_visual, results_visual], dim=2)
        visual = nn.functional.interpolate(visual, scale_factor=_scale)
        num_image = 10 if num_image > 10 else num_image
        image_grid = vutils.make_grid(visual, nrow=num_image, padding=2, scale_each=True)
        vutils.save_image(image_grid, PJ(segment_directory, f"{name}.jpg"), nrow=1)
