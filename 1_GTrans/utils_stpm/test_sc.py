import torch
import torch.nn as nn

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    x1 = torch.rand([64, 8, 4])
    x2 = torch.rand([64, 8, 4])
    x1, x2 = x1.to(device), x2.to(device)
    # net = model(x)
    similarity_loss = torch.nn.CosineSimilarity()
    cos = similarity_loss(x1, x2)
    print('cos_shape: ', cos.shape)
    print(cos)