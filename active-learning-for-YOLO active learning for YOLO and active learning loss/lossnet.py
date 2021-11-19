class LossNet(nn.Module):
    def __init__(self, basemodel, feature_sizes=[32], num_channels=[255], interm_dim=128):
        super(LossNet, self).__init__()
        self.base_model = base_model
        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.linear = nn.Linear(interm_dim, 1)
    
    def forward(self, features):
        out_p = self.base_model.forward(features)
        out1 = self.GAP1(features)
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        loss_pred = self.linear(out1)
        return out_p, loss_pred


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)
    
    return uncertainty.cpu()

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss