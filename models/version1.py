import time
import torch
from torch import nn
'''
class Version1_Loss:
    def __init__(self, weights_to = "ce", temp=3):
        self.weights_to = weights_to
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss()

        if self.weights_to == "ce":
            self.value = self.ce_weighted
        elif self.weights_to == "kld":
            self.value = self.kld_weighted
        elif self.weights_to == "both":
            self.value = self.both_weighted

    def ce_loss(self, outputs, labels):
        return self.ce(outputs, labels.long())

    def kld_loss(self, teacher_out, outputs):
        teacher_out_temp = nn.Softmax()(teacher_out / self.temp)
        outputs_temp = nn.LogSoftmax()(outputs / self.temp)

        kl = self.kl(outputs_temp, teacher_out_temp)*self.temp*self.temp
        return kl

    def ce_weighted(self, outputs, teacher_out, labels, weight):
        loss = self.kld_loss(teacher_out, outputs) + weight*self.ce_loss(outputs, labels)
        return loss

    def kld_weighted(self, outputs, teacher_out, labels, weight):
        loss = weight*self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels)
        return loss

    def both_weighted(self, outputs, teacher_out, labels, weight):
        loss = weight*(self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels))
        return loss
'''
class Version1_Loss:
    def __init__(self, weights_to = "ce", temp=3):
        self.weights_to = weights_to
        self.temp = temp
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kl = nn.KLDivLoss(reduction='none')

        if self.weights_to == "ce":
            self.value = self.ce_weighted
        elif self.weights_to == "kld":
            self.value = self.kld_weighted
        elif self.weights_to == "both":
            self.value = self.both_weighted

    def ce_loss(self, outputs, labels):
        return self.ce(outputs, labels.long())

    def kld_loss(self, teacher_out, outputs):
        teacher_out_temp = nn.Softmax()(teacher_out / self.temp)
        outputs_temp = nn.LogSoftmax()(outputs / self.temp)

        kl = self.kl(outputs_temp, teacher_out_temp)*self.temp*self.temp
        kl = torch.mean(kl, 1)
        return kl

    def ce_weighted(self, outputs, teacher_out, labels, weight):
        loss = self.kld_loss(teacher_out, outputs) + weight*self.ce_loss(outputs, labels)
        return loss.mean()

    def kld_weighted(self, outputs, teacher_out, labels, weight):
        loss = weight*self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels)
        return loss.mean()

    def both_weighted(self, outputs, teacher_out, labels, weight):
        loss = weight*(self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels))
        return loss.mean()

def training(teacher_model, student_model, dataloaders, loss_func, optimizer, scheduler, num_epochs=20, mu = 1.1):
    start_time = time.time()

    weight = 0.1

    for epoch in range(num_epochs):

        print('Epoch No. --> {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        print(weight)

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
              
                    loss = loss_func.value(outputs, teacher_out, labels, weight)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
      ### Updating Weights ###
        weight = mu*weight
        if weight>1:
            weight = 1
    time_taken = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_taken // 60, time_taken % 60))

    return student_model