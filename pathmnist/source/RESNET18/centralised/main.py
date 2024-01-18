import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataset
from models import CNNTarget
from utils import progress_bar

def train(epoch, net, train_loader, criterion, optimizer, device, writer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs = net(inputs)
        targets = targets.squeeze()
        print(targets.shape)
        loss = criterion(inputs, targets)
        loss.backward()
        optimizer.step()


        train_loss += loss.item()

        _, predicted = inputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        writer.add_scalar('Loss/train', train_loss/(batch_idx+1), epoch)
        writer.add_scalar('Accuracy/train', 100.*correct/total, epoch)


def test(epoch, net, test_loader, criterion, device, writer):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0

    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = net(inputs)
            targets = targets.squeeze()
            loss = criterion(inputs, targets)

            test_loss += loss.item()

            _, predicted = inputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            writer.add_scalar('Loss/test', test_loss/(batch_idx+1), epoch)
            writer.add_scalar('Accuracy/test', 100.*correct/total, epoch)




def run(data_flag, epoch, lr, batch_size, optimizer, device, writer):
    start_epoch = 0
    train_loader, train_loader_at_eval, test_loader, val_loader = get_dataset(data_flag, batch_size)
    
    net = CNNTarget()
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch, net, train_loader, criterion, optimizer, device, writer)
        # test(epoch, net, test_loader, criterion, device, writer)
        # scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', type=str, default='pathmnist', help='name of dataset')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')

    args = parser.parse_args()

    data_name = args.data_name
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    optimizer = args.optimizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()
    run(data_name, epochs, lr, batch_size, optimizer, device, writer)
