import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: 从其他地方copy来的DA_loss实现，先删掉现在的angle和scale，改成CMD
class DA_Loss(nn.Module):
    def __init__(self, tradeoff_angle, tradeoff_scale, treshold, device):
        super().__init__()
        self.tradeoff_angle = tradeoff_angle
        self.tradeoff_scale = tradeoff_scale
        self.treshold = treshold
        self.device = device
        
    def forward(self, H1, H2):
        # 对特征先做 L2 归一化，再计算协方差与距离
        H1 = F.normalize(H1, p=2, dim=1)
        H2 = F.normalize(H2, p=2, dim=1)

        b, p = H1.shape
        device = H1.device

        A = torch.cat((torch.ones(b, 1, device=device), H1), 1)
        B = torch.cat((torch.ones(b, 1, device=device), H2), 1)
        cov_A = (A.t() @ A)
        cov_B = (B.t() @ B)

        _, L_A, _ = torch.linalg.svd(cov_A)
        _, L_B, _ = torch.linalg.svd(cov_B)

        eigen_A = torch.cumsum(L_A.detach(), dim=0) / L_A.sum()
        eigen_B = torch.cumsum(L_B.detach(), dim=0) / L_B.sum()

        if (eigen_A[1] > self.treshold):
            T = eigen_A[1].detach()
        else:
            T = self.treshold

        index_A = torch.argwhere(eigen_A.detach() <= T)[-1]

        if (eigen_B[1] > self.treshold):
            T = eigen_B[1].detach()
        else:
            T = self.treshold

        index_B = torch.argwhere(eigen_B.detach() <= T)[-1]

        k = max(index_A, index_B)[0]

        A = torch.linalg.pinv(cov_A, rtol=(L_A[k] / L_A[0]).detach())
        B = torch.linalg.pinv(cov_B, rtol=(L_B[k] / L_B[0]).detach())

        cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        angle_dist = torch.dist(torch.ones((p + 1), device=device), (cos_sim(A, B)), p=1) / (p + 1)
        scale_dist = torch.dist((L_A[:k]), (L_B[:k])) / k

        loss = self.tradeoff_angle * angle_dist + self.tradeoff_scale * scale_dist
        return loss, angle_dist.detach(), scale_dist.detach()