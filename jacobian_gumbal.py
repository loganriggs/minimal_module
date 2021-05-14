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
from GumbalMaskWeights import GumbalMaskWeights
from mask_zero_one import gumbal
from torch.nn import functional as F
from matplotlib import colors
from functools import partial


def get_activations(network, x):
    activations = []
    hooks = []
    for name, m in network.named_modules():
        if isinstance(m, nn.Linear):
            save_activations = lambda mod, inp, out: activations.append(out)
            hooks.append(m.register_forward_hook(save_activations))

    network(x)
    for h in hooks:
        h.remove()

    return torch.hstack(activations)

#6 -> 1000 -> 2
#3 -> 1000 -> 1

class GumbalWeightMask(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.fc1 = GumbalMaskWeights(shape[0], shape[1])
        self.fc2 = GumbalMaskWeights(shape[1], shape[1])
        self.fc3 = GumbalMaskWeights(shape[1], shape[2])

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output

def modify_parameters(mask, model):
    index = 0
    for _, v in model.parameters():
        size_of_v = v.size()[0]
        # Change actual value pointed to by v to modified_parameters
        v.mul_(mask[index:index+size_of_v])
        index += size_of_v

def get_parameters(model):
    return torch.cat(list(model.parameters()))


def compute_adjacency_matrix(embedding):
    '''Compute correlation based adjacency matrix from embedding.'''
    centered_embedding = embedding - embedding.mean()
    centered_embedding = centered_embedding / centered_embedding.abs().mean()
    cov = centered_embedding @ centered_embedding.T
    inv_std = 1 / cov.diag().sqrt()
    corr = cov * inv_std * inv_std.unsqueeze(1)
    adj_mat = corr ** 2
    return adj_mat


def compute_laplacian(embedding):
    '''Computes normalized Laplacian from embedding.

    Parameters:
        embedding (Tensor): n_neurons × n_features

    Returns:
        lap (Tensor): n_neurons × n_neurons
    '''
    # Compute adjacency matrix
    adj_mat = compute_adjacency_matrix(embedding)

    # Compute normalized laplacian
    inv_sqrt_degree = 1 / adj_mat.sum(0, keepdims=True).sqrt()
    adjnorm = adj_mat * inv_sqrt_degree * inv_sqrt_degree.T
    lap = torch.eye(adjnorm.shape[0]).to(device) - adjnorm

    return lap

def train(args, model, train_loader, optimizer, epoch, mask_training = False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.L1Loss()(output, target)
        original_loss = loss.detach().cpu()
        alpha = 0.001
        mask_loss =  alpha * sum([weights_layer.exp().sum()
                     for name, weights_layer in model.named_parameters() if "logits" in name])
        loss += mask_loss
        #Jacobian
        batch_actloss = 0.0
        if(not mask_training):
            jacs = [torch.autograd.functional.jacobian(partial(get_activations, model), x) for x in data]
            jac = torch.cat(jacs, dim=1).to(device)

            # jac = jac.flatten(1)
            #(64, 1, 64, 3) --> (64, 192)
            #Q: why flatten from 4->2 dim?
            lap = compute_laplacian(jac)
            eigval, eigvec = torch.symeig(lap, eigenvectors=True)
            #Q: Is the Poisson prior below the # of k clusters?
            n_clusters = 2
            eigval = eigval[n_clusters-1]
            actloss = eigval
            if(actloss < 0 ):
                print("Eigenvalues: ", eigval < 0)
            assert torch.allclose(eigval, eigval.sort(descending=False)[0]), "Eigenvalues not in ascending order."
            # assert (torch.diff(eigval) > 1e-7).all(), "Non-distinct eigenvalues."
            #Q: I kept getting this error. What's the bad possibility here of them being equal within a precision?

            #Q: I made this loss way bigger, but that's because it's much tinier than others.
            alpha_jacobian = 100
            batch_actloss = alpha_jacobian*actloss.detach().cpu()
            loss += alpha_jacobian * actloss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f} | Mask_loss {:.3f} | jac_loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), original_loss, mask_loss, batch_actloss))
            if args.dry_run:
                break

def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        test_loss = torch.mean(torch.stack([torch.mean(nn.L1Loss()(model(data), target)) for data, target in test_loader]))
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))
    return test_loss

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
        if "logit" in name:
            precision_list.append(param.cpu().detach().clone())
    return precision_list


def train_test(args, optimizer, model, train_loader, test_loader, mask_training = False):
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch, mask_training= mask_training)
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
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--num-workers', type=int, default=0,
                    help='Number of workers for parallelization')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()


run = 1
if(run == 0): #Quick run
    args.epochs = 1
    num_workers = 2
elif(run == 1): #Debug run
    args.epochs = 1
    num_workers = 0
else: #Full Run
    args.epochs = 6
    num_workers = 2

# torch.manual_seed(args.seed)
device = torch.cuda if use_cuda else torch
device = "cuda" if use_cuda else "cpu"
# torch.set_default_tensor_type(device.FloatTensor)
print("cuda = ", device)
args.device = device

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


loss_results = np.zeros(6)
datatype = 1
if(datatype == 0): #Single add or mult
    data_name = "add_mult_single"
    labels = ["Single Experiment", "Add", "Mult"]
else: # Both add_mult and mult_add
    data_name = "add_mult"
    labels = ["Double Experiment", "(Add, Mult)", "(Mult, Add)"]


mult_add = np.load(data_name + "_data.npy")
mult_add_labels = np.load(data_name + "_labels.npy")
input_output_shape = [mult_add.shape[1], mult_add_labels.shape[1]]
mult_add = torch.Tensor(mult_add).to(device)
mult_add_labels = torch.Tensor(mult_add_labels).to(device)


train_dataset, test_dataset = train_test_dataset_maker(mult_add, mult_add_labels, test_percent = 0.3)
train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
test_loader  = torch.utils.data.DataLoader(test_dataset, **test_kwargs)


networkShape = [input_output_shape[0], 1000, input_output_shape[1]]
model = GumbalWeightMask(networkShape).to(device)


print("Beginning\n=================================")

optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

train_test(args, optimizer, model, train_loader, test_loader)
print("Original Training\n=====================================\n")
print_examples(model, test_loader)
precision_original = precision_parameter_to_list(model)

#This may not work: try
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

#Test original Model loss
loss_results[0] = test(model, test_loader_mult_add)
loss_results[1] = test(model, test_loader_add_mult)


#Reset optimizer learning rate
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

#Train add_mult
train_test(args, optimizer, model, train_loader_add_mult, test_loader_add_mult, mask_training=True)
precision_add_mult = precision_parameter_to_list(model)
print("Only add_mult: 0 , 1 should do well")
print_examples(model, test_loader)
print("Add_Mult should be lower loss")
print("Mult_add Loss: ")
loss_results[2] = test(model, test_loader_mult_add)
print("Add_Mult Loss")
loss_results[3] = test(model, test_loader_add_mult)


#Re-point optimzer to the original_model
optimizer_new = optim.Adadelta(original_model.parameters(), lr=args.lr)

#Train mult_add
train_test(args, optimizer_new, original_model, train_loader_mult_add, test_loader_mult_add, mask_training=True)
precision_mult_add = precision_parameter_to_list(original_model)
print("Only mult: 1 , 0 should do well")
print_examples(original_model, test_loader)
print("Mult_Add should be lower loss")
print("Mult_add Loss: ")
loss_results[4] = test(original_model, test_loader_mult_add)
print("Add_Mult Loss")
loss_results[5] = test(original_model, test_loader_add_mult)


print("====================Loss=================")
print("Case        | Mult Add       | Add Mult")
print("Original    | {0:.2f}         | {1:.2f}".format(loss_results[0], loss_results[1]))
print("Add_Mult    | {0:.2f}         | {1:.2f}".format(loss_results[2], loss_results[3]))
print("Mult_Add    | {0:.2f}         | {1:.2f}".format(loss_results[4], loss_results[5]))


if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")

print("Just masks themselves")
print("Layer |   Add %    |   Mult %  |   Add_size  |   Mult_size  | Original_size")
#Tuple w/ three values:
#"Name of layer", add %, Mult %
#A chart w/ the values as well, as we have it
num_of_layers = len(precision_original)
layer_name = []
weights_shared_add = [0]*num_of_layers
weights_shared_mult = [0]*num_of_layers
for name, _ in model.named_parameters():
    if "logit" in name:
        layer_name.append(name)


for layer_index in range(num_of_layers):
    average_weights_shared_add = 0
    average_weights_shared_mult = 0
    average_add_size = 0
    average_mult_size = 0
    average_original_size = 0
    #Loop N times since we're sampling from a distribution
    N = 100
    for n in range(N):
        add_mask = gumbal(precision_add_mult[layer_index])
        mult_mask = gumbal(precision_mult_add[layer_index])
        add_size = torch.count_nonzero(add_mask)
        mult_size = torch.count_nonzero(mult_mask)
        shared_size = torch.count_nonzero(add_mask*mult_mask)
        # if n == 0: #plot first mask
        #     encoding = add_mask + 2*mult_mask
        #     if(encoding.shape.__len__() == 1):
        #         encoding = encoding[None, :] #Extend dimension
        #     cmap = colors.ListedColormap(['black', 'blue', 'red', 'purple'])
        #     bounds = [0, 1, 2,3,4]
        #     norm = colors.BoundaryNorm(bounds, cmap.N)
        #     plt.figure(x)
        #     if x % 2 != 0: #odd is bias layer
        #         aspect_size = 100
        #     elif x == 0:
        #         aspect_size = 5
        #     elif x == 2:
        #         aspect_size = 1
        #     else:
        #         aspect_size = 10
        #     im = plt.imshow(encoding, cmap=cmap, norm=norm, aspect = aspect_size)
        #     plt.title("Layer " + str(x+1))
        #     plt.xlabel("Neuron")
        #     plt.ylabel("Similarity")
        #     im.axes.set_yticks([])

            #11 = shared (purple)
            #10 = add_mask only (red)
            #01 = mult_mask only (blue)
            #00 = not used by either (black)

        average_add_size += add_size
        average_mult_size += mult_size
        average_original_size += torch.count_nonzero(gumbal(precision_original[layer_index]))
        if(add_size != 0):
            average_weights_shared_add += shared_size / add_size
        if(mult_size != 0):
            average_weights_shared_mult += shared_size / mult_size
    average_weights_shared_add = average_weights_shared_add / N
    average_weights_shared_mult = average_weights_shared_mult / N
    average_add_size = average_add_size / N
    average_mult_size = average_mult_size / N
    average_original_size = average_original_size / N
    print(layer_index, " , ", average_weights_shared_add, ", ", average_weights_shared_mult, ", ", average_add_size, " , ", average_mult_size, " , ", average_original_size)
    weights_shared_add[layer_index] = average_weights_shared_add
    weights_shared_mult[layer_index] = average_weights_shared_mult

x = np.arange(len(layer_name))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, weights_shared_add, width, label=labels[1])
rects2 = ax.bar(x + width/2, weights_shared_mult, width, label=labels[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Percentage Weights Shared')
ax.set_title(labels[0])
ax.set_xticks(x)
ax.set_xticklabels(layer_name)
ax.legend()
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.ylim((0.0,1.0))

plt.show()