import torch

class TrF(torch.nn.Module):
    def __init__(self, model, criterion, num_classes):
        super(TrF, self).__init__()
        self.model = model
        self.criterion = criterion
        self.labels = torch.arange(num_classes).to(next(model.parameters()))

    def forward(self, y_pred):
        trace = 0.0
        prob = torch.nn.functional.softmax(y_pred, dim=1)
        idx_sampled = prob.multinomial(1)
        y_sampled = self.labels[idx_sampled].long().squeeze()
        # print(y_pred.dtype, y_sampled.dtype)
        loss = self.criterion(y_pred, y_sampled)
        loss.backward(retain_graph=True)
        for param in self.model.parameters():
            trace += (param.grad ** 2).sum()
        self.model.zero_grad()
        return trace
