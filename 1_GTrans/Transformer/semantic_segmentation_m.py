import copy

from utils import change_names

import torch
import torch.nn as nn

from visual_transformer import FilterBasedTokenizer, Transformer, Projector, RecurrentTokenizer, Transformer_Decoder
import torchvision.models as models
from resnet import modified_resnet50_2, modified_resnet18

class PanopticFPN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone = backbone
        # self.backbone = ResNet50Backbone()
        # self.backbone = VT_FPN()
        # load_resnet(self.backbone)

        self.skip_con_conv2 = nn.Conv2d(64, 1024, kernel_size=1)
        self.skip_con_conv3 = nn.Conv2d(128, 1024, kernel_size=1)
        self.skip_con_conv4 = nn.Conv2d(256, 1024, kernel_size=1)
        self.skip_con_conv5 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2)
        # self.ss_branch = SemanticSegmentationBranch(256,512,1024)

    def forward(self, c2, c3, c4):
        # c2, c3, c4, c5 = self.backbone(X)
        # c2, c3, c4 = self.backbone(X)
        # p5 = self.skip_con_conv5(c5) # [32, 256, 7, 7]
        # p4 = self.skip_con_conv4(c4) + self.upsample(p5) # [32, 256, 14, 14]
        p4 = self.skip_con_conv4(c4)
        p3 = self.skip_con_conv3(c3) + self.upsample(p4) # [32, 256, 28, 28]

        p2 = self.skip_con_conv2(c2) + self.upsample(p3) # [32, 256, 56, 56]
        # result = self.ss_branch(c2, c3, c4)
        # print(p2.shape, p3.shape, p4.shape, p5.shape)
        return p2, p3, p4


class VT_FPN(nn.Module):
    def __init__(self, n_visual_tokens=8, backbone_unpre=None, backbone_pre=None):
        super().__init__()
        self.n_visual_tokens = n_visual_tokens
        self.backbone_pre = backbone_pre
        self.backbone_unpre = backbone_unpre
        self.channel = 1024
        # self.backbone = ResNet50Backbone()
        # load_resnet(self.backbone)

        self.tokenizer2 = FilterBasedTokenizer(64, self.channel, n_visual_tokens)
        self.tokenizer3 = FilterBasedTokenizer(128, self.channel, n_visual_tokens)
        self.tokenizer4 = FilterBasedTokenizer(256, self.channel, n_visual_tokens)
        # self.tokenizer5 = FilterBasedTokenizer(2048, self.channel, n_visual_tokens)
        self.transformer = Transformer(self.channel)
        self.Decoder = Transformer_Decoder(self.channel, self.channel)
        self.projector2 = Projector(64, self.channel)
        self.projector3 = Projector(128, self.channel)
        self.projector4 = Projector(256, self.channel)
        # self.projector5 = Projector(2048, self.channel)

        self.upsample = nn.Upsample(scale_factor=2)
        # self.FPN = PanopticFPN()
        # self.noise = Noise_Add()
        # self.ss_branch = SemanticSegmentationBranch(256, 512, 1024, 2048)

    def forward(self, X):
        bs, ch, h, w = X.shape
        # c2, c3, c4, c5 = self.backbone(X)
        # Teacher_Feature
        ft1, ft2, ft3 = self.backbone_pre(X)
        c2, c3, c4 = self.backbone_unpre(X)
        # c2, c3, c4 = self.FPN(c2, c3, c4)
        # ft1, ft2, ft3 = self.backbone_pre(X_t)
        # c2, c3, c4 = self.backbone_unpre(X_s)
        # 注释：从第二维度开始展开，[bs, c, H, W] -> [bs, c, HW]
        c2, c3, c4 = torch.flatten(c2, start_dim=2), torch.flatten(c3, start_dim=2), torch.flatten(c4, start_dim=2)
        t2, t3, t4 = torch.flatten(ft1, start_dim=2), torch.flatten(ft2, start_dim=2), torch.flatten(ft3, start_dim=2)
        # 注释：c2-[32,64,3136]  c3-[32,128,784]  c4-[32,256,196]
        visual_tokens2 = self.tokenizer2(c2) # visual_tokens2:  torch.Size([32, 1024, 8])
        visual_tokens3 = self.tokenizer3(c3)
        visual_tokens4 = self.tokenizer4(c4)

        token_t2, token_t3, token_t4 = self.tokenizer2(t2), self.tokenizer3(t3), self.tokenizer4(t4)
        all_token_t = torch.cat((token_t2, token_t3, token_t4), dim=2)
        # 注释：经过Token之后：[32,1024,8]
        # visual_tokens5 = self.tokenizer5(c5)
        all_visual_tokens = torch.cat((visual_tokens2, visual_tokens3, visual_tokens4), dim=2) # [32, 1024, 8*3]
        encoder_s1 = self.transformer(all_visual_tokens)
        # encoder_s2 = self.transformer(encoder_s1)
        decoder_1 = self.Decoder(all_token_t, encoder_s1)
        # decoder_2 = self.Decoder(decoder_1, encoder_s2)
        t2, t3, t4 = torch.split(decoder_1, self.n_visual_tokens, dim=2)
        # t2, t3, t4 = torch.split(self.Decoder(all_visual_tokens, self.transformer(all_visual_tokens)), self.n_visual_tokens, dim=2)

        # p5 = self.projector5(c5, t5).view(bs, -1, h // 32, w // 32)
        p4 = self.projector4(c4, t4).view(bs, -1, h // 16, w // 16)
        p3 = self.projector3(c3, t3).view(bs, -1, h // 8, w // 8)
        p2 = self.projector2(c2, t2).view(bs, -1, h // 4, w // 4)

        return [ft1, ft2, ft3], [p2, p3, p4]



if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    backbone = modified_resnet18(pretrained=True).to(device)
    model = Noise_Add().to(device)
    x = torch.rand([4, 3, 224, 224])
    x = x.to(device)
    net = model(x)
    
