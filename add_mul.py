from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.model_selection import train_test_split
from GaussianNoise import GaussianNoise
from copy import deepcopy
import matplotlib.pyplot as plt

def modify_parameters(mask, model):
    index = 0
    for _, v in model.parameters():
        size_of_v = v.size()[0]
        # Change actual value pointed to by v to modified_parameters
        v.mul_(mask[index:index+size_of_v])
        index += size_of_v

def get_parameters(model):
    return torch.cat(list(model.parameters()))

def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.L1Loss()(output, target)
        loss += sum([module.precisionlog.exp().sum()
                     for module in model if isinstance(module, GaussianNoise)])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(target)))
            if args.dry_run:
                break

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        test_loss = torch.mean(torch.stack([torch.sum(nn.L1Loss()(model(data), target)) for data, target in test_loader]))
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss/1000)) #TODO change 1000 to the batch size

def print_examples(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss_temp = nn.L1Loss()
            loss = torch.sum(loss_temp(output, target))
            print_eight = 0
            for x in range(8):

            # for one_data, one_target in zip(output, target):
                print("Input: ", data[x].cpu().numpy())
                print("Predicted: ", output[x].cpu().numpy(), " == ", target[x].cpu().numpy())
                # print_eight += 1
                # if(print_eight > 8):
                #     break
            break

def print_variance(model):
    for name, param in model.named_parameters():
        if "variance" in name:
            print(name, " : ", torch.mean(param).cpu().detach().numpy(), " | max: ", torch.max(param).cpu().detach().numpy(), " | min: ", torch.min(param).cpu().detach().numpy())

def precision_parameter_to_list(model):
    precision_list = []
    for name, param in model.named_parameters():
        if "precisionlog" in name:
            precision_list.append(param.cpu().detach().clone())
    return precision_list

def train_test(args, optimizer, model, train_loader, test_loader):
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()

def train_test_dataset_maker(mult_add, mult_add_labels, test_percent = .33):
    x_train, x_test, y_train, y_test = train_test_split(mult_add, mult_add_labels, test_size=test_percent)
    train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test ,y_test)
    return train_dataset, test_dataset

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--num-workers', type=int, default=0,
                    help='Number of workers for parallelization')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()


quick_run = True
full_run = True
debug_run = False
if(quick_run):
    args.epochs = 1
    num_workers = 1
if(full_run):
    args.epochs = 20
    num_workers = 1
if(debug_run):
    args.epochs = 1
    num_workers = 0

# torch.manual_seed(args.seed)

device = torch.cuda if use_cuda else torch
# torch.set_default_tensor_type(device.FloatTensor)
print("cuda = ", device)

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': num_workers,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
# dataset1 = datasets.MNIST('../data', train=True, download=True,
#                    transform=transform)
# dataset2 = datasets.MNIST('../data', train=False,
#                    transform=transform)

#add_mul example
# (2,3,0,5,6,1)
# so (2+3, 5*6) is the label
# I can easily iterate a lot

mult_add = np.load("add_mult_data.npy")
mult_add_labels = np.load("add_mult_labels.npy")
mult_add = torch.Tensor(mult_add)
mult_add_labels = torch.Tensor(mult_add_labels)

train_dataset, test_dataset = train_test_dataset_maker(mult_add, mult_add_labels, test_percent = 0.3)
train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
test_loader  = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

model = nn.Sequential(
    GaussianNoise(6),
    nn.Linear(6, 400),
    GaussianNoise(400),
    nn.ReLU(),
    nn.Linear(400, 400),
    GaussianNoise(400),
    nn.ReLU(),
    nn.Linear(400, 2),
    GaussianNoise(2)
    )

print("Beginning\n=================================")
print_variance(model)
# torch.ones_like(model.parameters())

optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

train_test(args, optimizer, model, train_loader, test_loader)
precision_original = precision_parameter_to_list(model)

# Set gradient False for all layers (except GN)
for name, param in model.named_parameters():
    if "precisionlog" not in name:
        param.requires_grad = False

# Create subset datasets
# add_mult - first half
# mult_add - second half
half_index = mult_add.shape[0]//2
add_mult_only = mult_add[:half_index,:]
mult_add_only = mult_add[half_index:, :]

add_mult_only_labels = mult_add_labels[:half_index]
mult_add_only_labels = mult_add_labels[half_index:]

#Create data_loaders
train_dataset_mult_add, test_dataset_mult_add = train_test_dataset_maker(mult_add_only, mult_add_only_labels, test_percent = 0.3)
train_loader_mult_add = torch.utils.data.DataLoader(train_dataset_mult_add,**train_kwargs)
test_loader_mult_add  = torch.utils.data.DataLoader(test_dataset_mult_add, **test_kwargs)

train_dataset_add_mult, test_dataset_add_mult = train_test_dataset_maker(add_mult_only, add_mult_only_labels,  test_percent=0.3)
train_loader_add_mult = torch.utils.data.DataLoader(train_dataset_add_mult, **train_kwargs)
test_loader_add_mult = torch.utils.data.DataLoader(test_dataset_add_mult, **test_kwargs)

#Create deep copy of model
original_model = deepcopy(model)

#Train add_mult
train_test(args, optimizer, model, train_loader_add_mult, test_loader_add_mult)
precision_add_mult = precision_parameter_to_list(model)

#Re-point optimzer to the original_model
optimizer = optim.Adadelta(original_model.parameters(), lr=args.lr)

#Train mult_add
train_test(args, optimizer, original_model, train_loader_mult_add, test_loader_mult_add)
precision_mult_add = precision_parameter_to_list(original_model)

if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")

m = deepcopy(precision_original)
a = deepcopy(precision_original)
for x in range(4):
    m[x] = precision_mult_add[x] - precision_original[x]
    a[x] = precision_add_mult[x] - precision_original[x]
    plt.figure(x)
    plt.scatter(m[x], a[x])
    plt.title("Layer " + str(x))