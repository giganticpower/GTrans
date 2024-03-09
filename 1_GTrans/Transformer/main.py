import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from datasets.mvtec import MVTecDataset
from utils_stpm.util import time_string, convert_secs2time, AverageMeter
from utils_stpm.functions import cal_anomaly_maps, cal_loss
from utils_stpm.visualization import plt_fig
from semantic_segmentation_m import VT_FPN
from resnet import modified_resnet50_2, modified_resnet18, modified_resnet34
import visdom
import math


class STPM():
    def __init__(self, args):
        self.device = args.device
        self.data_path = args.data_path
        self.obj = args.obj
        self.img_resize = args.img_resize
        self.img_cropsize = args.img_cropsize
        self.validation_ratio = args.validation_ratio
        self.num_epochs = args.num_epochs
        self.trans_epochs = args.trans_epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.vis = args.vis
        self.model_dir = args.model_dir
        self.img_dir = args.img_dir
        self.lrf = 0.1
        self.decay_rate = 0.9

        self.load_model()
        self.load_dataset()
        self.kernel = np.ones((2, 2), np.uint8)

        # self.criterion = torch.nn.MSELoss(reduction='sum')
        # self.optimizer = torch.optim.SGD(self.model_s.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)  # 原来的momentum=0.9, weight_decay=0.0001
        self.optimizer = torch.optim.Adam(self.model_s.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0001)
        # self.lf = lambda x:((1 + math.cos(x * math.pi / args.num_epochs)) / 2) * (1 - self.lrf) + self.lrf
        self.lf = lambda x: self.decay_rate ** ( x / args.num_epochs)
        # self.lf = lambda x: ((1 + ((x / args.num_epochs)**2 - 2 * (x / args.num_epochs) + 1) / 2)) * (1 - self.lrf) + self.lrf
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def load_dataset(self):
        kwargs = {'num_workers': 0, 'pin_memory': False} if torch.cuda.is_available() else {}
        train_dataset = MVTecDataset(self.data_path, class_name=self.obj, is_train=True, resize=self.img_resize,
                                     cropsize=self.img_cropsize)
        img_nums = len(train_dataset)
        valid_num = int(img_nums * self.validation_ratio)
        train_num = img_nums - valid_num
        train_data, val_data = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, **kwargs)

    def load_model(self):
        self.model_t = modified_resnet18(pretrained=True).to(self.device)
        self.model_unpre = modified_resnet18(pretrained=False).to(self.device)
        self.model_s = VT_FPN(backbone_unpre=self.model_unpre, backbone_pre = self.model_t).to(self.device)

        for param in self.model_t.parameters():
            param.requires_grad = False  # False:屏蔽预训练模型的权重
        self.model_t.eval()

    def train(self):
        self.model_s.train()
        best_score = None
        best_epoch = None
        best_val_score = None
        start_time = time.time()
        epoch_time = AverageMeter()
        for epoch in range(1, self.num_epochs + 1):
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * ((self.num_epochs + 1) - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
            print('{:3d}/{:3d} --{}-- [{:s}] {:s}'.format(epoch, self.num_epochs, self.obj, time_string(), need_time))
            losses = AverageMeter()
            for (data, _, _) in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    features_t, features_s = self.model_s(data)
                    loss = cal_loss(features_s, features_t)
                    losses.update(loss.sum().item(), data.size(0))
                    loss.backward()
                    self.optimizer.step()

            # if (epoch == self.num_epochs):
            #     f.write('\nTrain Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))
            print('Train Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))
            vis.line(Y=np.array([losses.avg]), X=np.array([epoch]),
                     win=('%s_train_loss' % args.obj),
                     opts=dict(title='%s_train_loss' % args.obj),
                     update='append' if epoch > 0 else None
                     )

            val_loss = self.val(epoch)
            if best_val_score is None:
                best_val_score = val_loss
                self.save_checkpoint_score()
            elif val_loss < best_val_score:
                best_val_score = val_loss
                self.save_checkpoint_score()

            if epoch > self.num_epochs - 50:
                self.save_checkpoint_val()
                val_score = self.test()

                if best_score is None:
                    best_score = val_score
                    self.save_checkpoint()
                elif val_score >= best_score:
                    best_score = val_score
                    best_epoch = epoch
                    self.save_checkpoint()
                print('val_score: {:.4f}; best_score({}): {:.4f}'.format(val_score, best_epoch, best_score))

            epoch_time.update(time.time() - start_time)
            start_time = time.time()

        # print('Training end.')

    def val(self, epoch):
        self.model_s.eval()
        losses = AverageMeter()
        for (data, _, _) in self.val_loader:
            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t, features_s = self.model_s(data)
                loss = cal_loss(features_s, features_t)
                losses.update(loss.item(), data.size(0))
        self.scheduler.step()

        # if (epoch == self.num_epochs):
        #     f.write('\nVal Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))
        print('Val Epoch: {} loss: {:.6f}'.format(epoch, losses.avg))
        vis.line(Y=np.array([losses.avg]), X=np.array([epoch]),
                 win=('%s_val_loss' % args.obj),
                 opts=dict(title='%s_val_loss' % args.obj),
                 update='append' if epoch > 0 else None
                 )
        return losses.avg

    def save_checkpoint(self):
        state = {'model': self.model_s.state_dict()}
        torch.save(state, os.path.join(self.model_dir, 'model_s.pth'))

    def save_checkpoint_val(self):
        state = {'model': self.model_s.state_dict()}
        torch.save(state, os.path.join(self.model_dir, 'model_val.pth'))

    def save_checkpoint_score(self):
        state = {'model': self.model_s.state_dict()}
        torch.save(state, os.path.join(self.model_dir, 'model_score.pth'))

    def test(self):
        try:
            checkpoint = torch.load(os.path.join(self.model_dir, 'model_val.pth'))
        except:
            raise Exception('Check saved model path.')
        self.model_s.load_state_dict(checkpoint['model'])
        self.model_s.eval()

        kwargs = {'num_workers': 0, 'pin_memory': False} if torch.cuda.is_available() else {}
        test_dataset = MVTecDataset(self.data_path, class_name=self.obj, is_train=False, resize=self.img_resize,
                                    cropsize=self.img_cropsize)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)

        scores = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []
        feature_vis = []
        print('Testing...')
        inference_time = AverageMeter()
        for (data, label, mask) in tqdm(test_loader):
            start_time = time.time()
            test_imgs.extend(data.cpu().numpy())
            gt_list.extend(label.cpu().numpy())
            gt_mask_list.extend(mask.squeeze().cpu().numpy())

            data = data.to(self.device)
            with torch.set_grad_enabled(False):
                features_t, features_s = self.model_s(data)
                score, feature_map = cal_anomaly_maps(features_s, features_t, self.img_cropsize)
            inference_time.update(time.time() - start_time)
            feature_vis.extend(feature_map)
            scores.extend(score)

        # FPS
        fps_avg = 1 / inference_time.avg
        print('FPS: %.3f' % (fps_avg))
        fps.append(fps_avg)

        scores = np.asarray(scores)
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        image_score.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))

        precision, recall, thresholds = precision_recall_curve(gt_list.flatten(), img_scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        cls_threshold = thresholds[np.argmax(f1)]

        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        seg_threshold = thresholds[np.argmax(f1)]

        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        pixel_score.append(per_pixel_rocauc)

        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))
        f.write('%s: image ROCAUC: %.3f   pixel ROCAUC: %.3f\n' % (self.obj, img_roc_auc, per_pixel_rocauc))

        if self.vis:
            plt_fig(test_imgs, scores, img_scores, feature_vis, gt_mask_list, seg_threshold, cls_threshold,
                    self.img_dir, self.obj)

        return 0.5 * (img_roc_auc + per_pixel_rocauc)


def get_args():
    parser = argparse.ArgumentParser(description='STPM anomaly detection')
    parser.add_argument('--phase', default='train')
    parser.add_argument("--data_path", type=str,
                        default="/data/chh/yst/Anomaly_Detection/Knowledge_Distillation_AD_VGG/KD_AD-main/Dataset/MVTec")
    parser.add_argument('--obj', type=str, default='transistor')
    parser.add_argument('--img_resize', type=int, default=256)
    parser.add_argument('--img_cropsize', type=int, default=224)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--num_epochs', default=500)
    parser.add_argument('--trans_epochs', default=1)
    parser.add_argument('--lr', default=0.0005)  # default=0.4
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--vis', type=eval, choices=[True, False], default=False)
    parser.add_argument("--save_path", type=str, default="./mvtec_b1_index0.9_500_decoder_1layer_0.0005")
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    vis = visdom.Visdom(env='Test')
    CLASS_NAMES = ['capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
                     'carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable']

    image_score = []
    pixel_score = []
    fps = []

    f = open('output.txt', 'a')
    f.write('\n#------ base_model_index: 01*12 ------#\n')
    for class_name in CLASS_NAMES:
        print('###-------- current class: {} --------###'.format(class_name))
        args.obj = class_name
        args.model_dir = args.save_path + '/models' + '/' + args.obj
        # print(args.model_dir)
        args.img_dir = args.save_path + '/imgs' + '/' + args.obj
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.img_dir, exist_ok=True)

        stpm = STPM(args)
        if args.phase == 'train':
            stpm.train()
            print('#--- current class: {} ---#'.format(class_name))
            stpm.test()
        elif args.phase == 'test':
            stpm.test()
        else:
            print('Phase argument must be train or test.')

    image = np.mean(image_score)
    pixel = np.mean(pixel_score)
    fps = np.mean(fps)
    print('FPS: %.3f' % (fps))
    print('avg_image_score: %.4f' % (image))
    print('avg_pixel_score: %.4f' % (pixel))
    f.write('FPS: %.3f   image ROCAUC: %.4f   pixel ROCAUC: %.4f\n' % (fps, image, pixel))
    f.close()








