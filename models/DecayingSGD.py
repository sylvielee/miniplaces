from torch.optim import SGD

class DecayingSGD(SGD):
    """
    Implements classic SGD optimizer but with a gradual learning rate warmup 
    as described in:
    https://arxiv.org/pdf/1706.02677.pdf
    """
    def __init__(self, params, start_lr, k, num_epochs, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        self.epoch = 1
        self.num_epochs = num_epochs
        self.k = k
        self.start_lr = start_lr
        self.lr = start_lr
        super(DecayingSGD, self).__init__(params, start_lr, momentum, dampening, weight_decay, nesterov)

    def setlr(self):
        for group in self.param_groups:
            group['lr'] = self.lr

    def epoch_step(self):
        self.epoch += 1
        halfway = self.num_epochs/2 # purposefully integer division
        if self.epoch <= halfway:
            self.lr += (1./halfway)*self.start_lr
            self.setlr()