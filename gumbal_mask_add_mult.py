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
from GumbalMask import GumbalMask
from mask_zero_one import gumbal
from matplotlib import colors

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
        alpha = 1
        loss += alpha /sum([module.logits.exp().sum()
                     for module in model if isinstance(module, GumbalMask)])
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
    with torch.no_grad():
        test_loss = torch.mean(torch.stack([torch.mean(nn.L1Loss()(model(data), target)) for data, target in test_loader]))
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))

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

def precision_parameter_to_list(model, parameter_name="precisionlog"):
    precision_list = []
    for name, param in model.named_parameters():
        if parameter_name in name:
            precision_list.append(param.cpu().detach().clone())
    return precision_list

# TODO: ummmmm..... do we return the mask (which is sampled every time (right?)) or do we run it a few times and average?s
# def mask_parameter_to_list(model):
#     precision_list = []
#     for name, param in model.named_parameters():
#         if "logits" in name:
#             precision_list.append(param.cpu().detach().clone())
#     return precision_list

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
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--num-workers', type=int, default=0,
                    help='Number of workers for parallelization')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
print("CUDA: ", torch.cuda.is_available())


run = 2
if(run == 0): #Quick run
    args.epochs = 1
    num_workers = 1
elif(run == 1): #Debug run
    args.epochs = 1
    num_workers = 0
else: #Full Run
    args.epochs = 10
    num_workers = 1

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

datatype = 1
if(datatype == 0): #Single add or mult
    data_name = "add_mult_single"
else: # Both add_mult and mult_add
    data_name = "add_mult"

mult_add = np.load(data_name + "_data.npy")
mult_add_labels = np.load(data_name + "_labels.npy")
input_output_shape = [mult_add.shape[1], mult_add_labels.shape[1]]
mult_add = torch.Tensor(mult_add)
mult_add_labels = torch.Tensor(mult_add_labels)


train_dataset, test_dataset = train_test_dataset_maker(mult_add, mult_add_labels, test_percent = 0.3)
train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
test_loader  = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

nn_size = 400
model = nn.Sequential(
    # GumbalMask(input_output_shape[0]),
    nn.Linear(input_output_shape[0], nn_size),
    GumbalMask(nn_size),
    nn.ReLU(),
    nn.Linear(nn_size, nn_size),
    GumbalMask(nn_size),
    nn.ReLU(),
    nn.Linear(nn_size, input_output_shape[1]),
    # GumbalMask(input_output_shape[1])
    )

print("Beginning\n=================================")
print_variance(model)
# torch.ones_like(model.parameters())

optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

train_test(args, optimizer, model, train_loader, test_loader)
precision_original = precision_parameter_to_list(model, parameter_name="logits")
print("Original Training\n=====================================\n")
print_examples(model, test_loader)

# Set gradient False for all layers (except GN)
for name, param in model.named_parameters():
    if "logits" not in name:
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

#Reset optimizer learning rate
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

#Train add_mult
train_test(args, optimizer, model, train_loader_add_mult, test_loader_add_mult)
precision_add_mult = precision_parameter_to_list(model, parameter_name="logits")
print("Only add_mult: 0 , 1 should do well")
print_examples(model, test_loader)
print("Add_Mult should be lower loss")
print("Mult_add Loss: ")
test(model, test_loader_mult_add)
print("Add_Mult Loss")
test(model, test_loader_add_mult)


#Re-point optimzer to the original_model
optimizer = optim.Adadelta(original_model.parameters(), lr=args.lr)

#Train mult_add
train_test(args, optimizer, original_model, train_loader_mult_add, test_loader_mult_add)
precision_mult_add = precision_parameter_to_list(original_model, parameter_name="logits")
print("Only mult: 1 , 0 should do well")
print_examples(original_model, test_loader)
print("Mult_Add should be lower loss")
print("Mult_add Loss: ")
test(original_model, test_loader_mult_add)
print("Add_Mult Loss")
test(original_model, test_loader_add_mult)


if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")

m = deepcopy(precision_original)
a = deepcopy(precision_original)
print("Masks - Original")
for x in range(2):
    m[x] = precision_mult_add[x] - precision_original[x]
    a[x] = precision_add_mult[x] - precision_original[x]
    average_weights_shared = 0
    #Loop N times since we're sampling from a distribution
    N = 100
    for _ in range(N):
        average_weights_shared += torch.count_nonzero(gumbal(m[x]) == gumbal(m[x])) / precision_mult_add[x].shape[0]
    average_weights_shared = average_weights_shared / N
    print("Layer ", x, ": ", average_weights_shared, "% shared")
    # plt.figure(x)
    # plt.scatter(m[x], a[x])
    # plt.title("Layer " + str(x))

print("Just masks themselves")
print("Layer |   Add %    |   Mult %  |   Add_size  |   Mult_size  | Original_size")

fig = plt.figure()
plt.title("Single Add Mult")
for x in range(2):
    average_weights_shared_add = 0
    average_weights_shared_mult = 0
    average_add_size = 0
    average_mult_size = 0
    average_original_size = 0
    #Loop N times since we're sampling from a distribution
    N = 100
    for n in range(N):
        add_mask = gumbal(precision_add_mult[x])
        mult_mask = gumbal(precision_mult_add[x])
        add_mask_ind = torch.where(add_mask == 1)[0]
        mult_mask_ind = torch.where(mult_mask == 1)[0]
        add_size = add_mask_ind.shape[0]
        mult_size = mult_mask_ind.shape[0]
        shared_size = len([1 for x,y in zip(add_mask, mult_mask) if x==1 and y == 1])
        if(n ==0): #plot first mask
            encoding = add_mask + 2*mult_mask
            encoding = encoding[None, :] #Extend dimension
            cmap = colors.ListedColormap(['black', 'blue', 'red', 'purple'])
            bounds = [0, 1, 2,3,4]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            plt.subplot(2,1,x+1)
            im = plt.imshow(encoding, cmap=cmap, norm=norm, aspect = 100)
            plt.title("Layer " + str(x+1))
            plt.xlabel("Neuron")
            plt.ylabel("Similarity")
            im.axes.set_yticks([])

            #11 = shared (purple)
            #10 = add_mask only (red)
            #01 = mult_mask only (blue)
            #00 = not used by either (black)

        average_add_size += add_size
        average_mult_size += mult_size
        average_original_size += torch.count_nonzero(gumbal(precision_original[x]))
        if(add_size != 0):
            average_weights_shared_add += shared_size / add_size
        if(mult_size != 0):
            average_weights_shared_mult += shared_size / mult_size
    average_weights_shared_add = average_weights_shared_add / N
    average_weights_shared_mult = average_weights_shared_mult / N
    average_add_size = average_add_size / N
    average_mult_size = average_mult_size / N
    average_original_size = average_original_size / N
    print(x, " , ", average_weights_shared_add, ", ", average_weights_shared_mult, ", ", average_add_size, " , ", average_mult_size, " , ", average_original_size)

