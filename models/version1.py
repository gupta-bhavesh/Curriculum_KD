import time
import torch
from torch import nn
import copy
import torch.nn.functional as F
'''
class Version1_Loss:
    def __init__(self, weights_to = "ce", alpha=0.3, temp=3):
        self.weights_to = weights_to
        self.temp = temp
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss()
        print('Version 1', weights_to)
        if self.weights_to == "ce":
            self.value = self.ce_weighted
        elif self.weights_to == "kld":
            self.value = self.kld_weighted
        elif self.weights_to == "both":
            self.value = self.both_weighted

    def ce_loss(self, outputs, labels):
        return self.ce(outputs, labels.long())

    def kld_loss(self, teacher_out, outputs):
        teacher_out_temp = F.softmax(teacher_out / self.temp, dim=1)
        outputs_temp = F.log_softmax(outputs/self.temp, dim=1)

        kl = self.kl(outputs_temp, teacher_out_temp)*self.temp*self.temp
        return kl

    def ce_weighted(self, outputs, teacher_out, labels, weight):
        loss = self.alpha*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*weight*self.ce_loss(outputs, labels)
        return loss

    def kld_weighted(self, outputs, teacher_out, labels, weight):
        loss = self.alpha*weight*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.ce_loss(outputs, labels)
        return loss

    def both_weighted(self, outputs, teacher_out, labels, weight):
        loss = weight*(self.alpha*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.ce_loss(outputs, labels))
        return loss

'''
class Version1_Loss:
    def __init__(self, weights_to = "ce", alpha=0.3, temp=3):
        self.weights_to = weights_to
        self.alpha = alpha
        self.temp = temp
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kl = nn.KLDivLoss(reduction='none')
        print('Version 1 Pixelwise', weights_to, temp, alpha)

        if self.weights_to == "ce":
            self.value = self.ce_weighted
        elif self.weights_to == "kld":
            self.value = self.kld_weighted
        elif self.weights_to == "both":
            self.value = self.both_weighted

    def ce_loss(self, outputs, labels):
        return self.ce(outputs, labels.long())

    def kld_loss(self, teacher_out, outputs):
        teacher_out_temp = F.softmax(teacher_out / self.temp, dim=1)
        outputs_temp = F.log_softmax(outputs/self.temp, dim=1)

        kl = self.kl(outputs_temp, teacher_out_temp)*self.temp*self.temp
        kl = torch.sum(kl, 1)
        return kl

    def ce_weighted(self, outputs, teacher_out, labels, weight):
        loss = self.alpha*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*weight*self.ce_loss(outputs, labels)
        return loss.mean()

    def kld_weighted(self, outputs, teacher_out, labels, weight):
        loss = self.alpha*weight*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.ce_loss(outputs, labels)
        return loss.mean()

    def both_weighted(self, outputs, teacher_out, labels, weight):
        loss = weight*(self.alpha*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.ce_loss(outputs, labels))
        return loss.mean()

def training(teacher_model, student_model, dataloaders, loss_func, optimizer, scheduler, num_epochs=20, mu = 1.1):
    start_time = time.time()

    weight = 0.05
    best_model_wts = copy.deepcopy(student_model.state_dict())
    best_loss = 50.0
    all_losses = {'train':[],'val':[]}

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
                    eval_loss = loss_func.ce_loss(outputs, labels).mean()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += eval_loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            all_losses[phase].append(epoch_loss)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(student_model.state_dict())

      ### Updating Weights ###
        weight = mu*weight
        if weight>1:
            weight = 1

    time_taken = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_taken // 60, time_taken % 60))
    
    print('Best val Loss: {:4f}'.format(best_loss))

    student_model.load_state_dict(best_model_wts)
    return student_model, all_losses