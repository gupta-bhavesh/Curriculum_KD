import time
import torch
from torch import nn

class Version234_Loss:
    def __init__(self, temp=3, version='Version 2'):
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss()

        if version =='Version 2':
            self.value = self.version2
        elif version =='Version 3':
            self.value = self.version3
        elif version =='Version 4':
            self.value = self.version4

    def ce_loss(self, outputs, labels):
        # print("CE", self.ce(outputs, labels.long()))
        return self.ce(outputs, labels.long())

    def kld_loss(self, teacher_out, outputs):
        teacher_out_temp = nn.Softmax()(teacher_out / self.temp)
        outputs_temp = nn.LogSoftmax()(outputs / self.temp)

        kl = self.kl(outputs_temp, teacher_out_temp)*self.temp*self.temp
        # print("KL", self.kl(outputs_temp, teacher_out_temp))
        return kl

    def version2(self, outputs, teacher_out, labels, mu):
        loss = mu * self.kld_loss(teacher_out, outputs) * self.ce_loss(outputs, labels)
        return loss

    def version3(self, outputs, teacher_out, labels, mu):
        loss = mu * (1-self.kld_loss(teacher_out, outputs)) * self.ce_loss(outputs, labels)
        # print(self.kld_loss(teacher_out, outputs), self.ce_loss(outputs, labels))
        return loss
    
    def version4(self, outputs, teacher_out, labels, mu):
        weight = 1 + (mu * (1-self.kld_loss(teacher_out, outputs)))
        loss = weight * self.ce_loss(outputs, labels)
        return loss

def training(teacher_model, student_model, dataloaders, loss_func, optimizer, scheduler, num_epochs=20):
    start_time = time.time()

    mu = 0.1

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

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        ### Updating mu such that it goes to 1 in half num of total epochs

        mu += 0.9/((num_epochs/2)-1)
        if mu>1:
            mu = 1

    time_taken = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_taken // 60, time_taken % 60))

    return student_model