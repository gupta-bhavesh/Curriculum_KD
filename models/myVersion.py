import time
import torch
from torch import nn
import copy
import torch.nn.functional as F
"""
Normal Loss
Sigmoid Loss
Normalized Loss

AP
GP
Decreasing AP
"""
class MyVersion_Loss:
    def __init__(self, weights_to = "none", alpha=0.5, temp=3):
        self.weights_to = weights_to
        self.alpha = alpha
        self.temp = temp
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kl = nn.KLDivLoss(reduction='none')
        print("MyVersion", temp, alpha, weights_to)
        if self.weights_to == "ce":
            self.value = self.ce_weighted
        if self.weights_to == "kld":
            self.value = self.kld_weighted
        if self.weights_to == "both":
            self.value = self.both_weighted

    def get_weights(self, teacher_out, labels, mu):
        teacher_loss = self.ce(teacher_out, labels.long())
        teacher_loss /= teacher_loss.max()

        weights = torch.ones(labels.shape, dtype=torch.float64).cuda()*(min(1,mu+0.1))
        weights[teacher_loss <= mu] = 1
        return weights

    def ce_loss(self, outputs, labels):
        return self.ce(outputs, labels.long())

    def kld_loss(self, teacher_out, outputs):
        teacher_out_temp = F.softmax(teacher_out / self.temp, dim=1)
        outputs_temp = F.log_softmax(outputs/self.temp, dim=1)

        kl = self.kl(outputs_temp, teacher_out_temp)*self.temp*self.temp
        kl = torch.mean(kl, 1)
        return kl

    def ce_weighted(self, outputs, teacher_out, labels, mu):
        loss = self.alpha*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.get_weights(teacher_out, labels, mu)*self.ce_loss(outputs, labels)
        return loss.mean()

    def kld_weighted(self, outputs, teacher_out, labels, mu):
        loss = self.alpha*self.get_weights(teacher_out, labels, mu)*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.ce_loss(outputs, labels)
        return loss.mean()

    def both_weighted(self, outputs, teacher_out, labels, mu):
        loss = self.get_weights(teacher_out, labels, mu)*(self.alpha*self.kld_loss(teacher_out, outputs) + (1-self.alpha)*self.ce_loss(outputs, labels))
        return loss.mean()

def training(teacher_model, student_model, dataloaders, loss_func, optimizer, scheduler, num_epochs=20):
    start_time = time.time()

    mu = 0.1
    # b = 0.18
    # d = 0.02
    best_model_wts = copy.deepcopy(student_model.state_dict())
    best_loss = 50.0
    all_losses = {'train':[],'val':[]}

    for epoch in range(num_epochs):

        print('Epoch No. --> {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        print(mu)

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
                    
                    loss = loss_func.value(outputs, teacher_out, labels, mu)
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

        ### Updating mu such that it goes to 1 in half num of total epochs

        # if epoch<10:
        #     mu += b - ((epoch)*d)
        # else:
        #     mu = 1
        mu+=0.1
        # mu = mu*1.2
        # mu = min(mu,1)

    time_taken = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_taken // 60, time_taken % 60))

    print('Best val Loss: {:4f}'.format(best_loss))
    student_model.load_state_dict(best_model_wts)

    return student_model, all_losses