import time
import torch
from torch import nn

class CurricullumKD_Loss:
    def __init__(self, weights_to = "none", alpha=0.5, temp=3):
        self.weights_to = weights_to
        self.alpha = alpha
        self.temp = temp
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kl = nn.KLDivLoss(reduction='none')

        if self.weights_to == "none":
            self.value = self.curricullum
        if self.weights_to == "ce":
            self.value = self.ce_weighted
        if self.weights_to == "kld":
            self.value = self.kld_weighted
        if self.weights_to == "both":
            self.value = self.both_weighted

    def get_weights(self, teacher_out, labels, mu):
        out_confidence = nn.Softmax()(teacher_out)
        weights = torch.ones(labels.shape, dtype=torch.float64).cuda()*(mu)
        weights[torch.logical_and(out_confidence[:,1,:,:] >= 1-mu, labels==1)] = 1
        weights[torch.logical_and(out_confidence[:,0,:,:] >= 1-mu, labels==0)] = 1
        return weights

    def ce_loss(self, outputs, labels):
        return self.ce(outputs, labels.long())

    def kld_loss(self, teacher_out, outputs):
        teacher_out_temp = nn.Softmax()(teacher_out / self.temp)
        outputs_temp = nn.LogSoftmax()(outputs / self.temp)

        kl = self.kl(outputs_temp, teacher_out_temp)*self.temp*self.temp
        kl = torch.mean(kl, 1)
        return kl

    def curricullum(self, outputs, teacher_out, labels):
        loss = self.alpha*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.ce_loss(outputs, labels)
        return loss.mean()

    def ce_weighted(self, outputs, teacher_out, labels, mu):
        loss = self.alpha*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.get_weights(teacher_out, labels, mu)*self.ce_loss(outputs, labels)
        return loss.mean()

    def kld_weighted(self, outputs, teacher_out, labels, mu):
        loss = self.alpha*self.get_weights(teacher_out, labels, mu)*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.ce_loss(outputs, labels)
        return loss.mean()

    def both_weighted(self, outputs, teacher_out, labels, mu):
        loss = self.get_weights(teacher_out, labels, mu)*(self.alpha*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.ce_loss(outputs, labels))
        return loss.mean()

def training(teacher_model, student_model, dataloaders, loss_func, optimizer, scheduler, num_epochs=20, temp=3):
    start_time = time.time()

    for epoch in range(num_epochs):

        print('Epoch No. --> {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        teacher_model.eval()
        for phase in ['train', 'val']:
            if phase == 'train':
                student_model.train()
            else:
                student_model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_out = teacher_model(inputs)
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = student_model(inputs)

                    loss = loss_func.value(outputs, teacher_out, labels, (epoch+1)/num_epochs)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

    time_taken = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_taken // 60, time_taken % 60))

    return student_model