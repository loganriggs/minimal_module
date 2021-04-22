import numpy as np

range_numbers = 200
total_dataset = range_numbers**2
index = 0

#Initialize values
add = np.zeros(shape=(4, total_dataset))
# add_label = np.zeros(shape=(total_dataset))
mult = np.zeros(shape=(4, total_dataset))
# mult_label = np.zeros(shape=(total_dataset))

#Enumerate all possible values
for x in range(range_numbers):
    for y in range(range_numbers):
        add[:, index] = [x,y,0,x+y]
        # add_label[index] = x+y
        mult[:, index] = [x,y,1,x*y]
        # mult_label[index] = x*y
        index += 1

#Shuffle add and mult
np.random.shuffle(add.T)
np.random.shuffle(mult.T)

#Convert into integers
add = add.astype(int)
mult = mult.astype(int)
# print(add)

#Split into inputs and labels
add_input = add[0:3,:]
add_label = add[3:4,:]
mult_input = mult[0:3,:]
mult_label = mult[3:4,:]

#Make Add_Null dataset
# add_mult
# input: 1,2,+, 3,4, *
# output: 3, 12

# 1: add_null
# input: 1,2,+, 0,0,0
# output: 3, 0

# 2: null_add
# 3: mult_null
# 4: null_mult


# add_mult | mult_add vs add_mult      [The smaller difference == more modular; i.e. reusability]
# add | mult (3 inputs) vs add vs mult [The bigger difference == more modular; i.e. different tasks have different modules]

'''
Algorithm:
1. train normally
2. Add regularizer/noise up to epsilon performance drop (can combine w/ 1)
3. Restrict dataset to [add/mult/ etc], and compare regularizer loss across mask of each neuron (w/ minimal epsilon perf drop)
4 (optional) same but w/ probabilistic mask
'''

#Combine add and mult
add_mult = np.concatenate((add_input, mult_input), axis=0)
add_mult_labels = np.concatenate((add_label, mult_label), axis=0)

print("Add mult shape: (6, " + str(total_dataset) + ") == ", add_mult.shape)
print("Add mult labels shape: ", add_mult_labels.shape)

#Combine mult and add
mult_add = np.concatenate((mult_input, add_input), axis=0)
mult_add_labels = np.concatenate((mult_label, add_label), axis=0)

#combine both
add_mult_both = np.concatenate((add_mult, mult_add), axis=1)
add_mult_both_labels = np.concatenate((add_mult_labels, mult_add_labels), axis=1)


np.save("add_mult_data", add_mult_both.T)
np.save("add_mult_labels", add_mult_both_labels.T)


#Verification: Deprecated
verify = False
if(verify):
    for pair, answer in zip(add_input.T, add_label):
        try:
            assert(pair[0]+pair[1] == answer)
        except AssertionError:
            print("Fail!")
            print("{0} + {1} = {2}".format(pair[0], pair[1], answer))
    for pair, answer in zip(mult_input.T, mult_label):
        try:
            assert(pair[0]*pair[1] == answer)
        except AssertionError:
            print("Fail!")
            print("{0} * {1} = {2}".format(pair[0], pair[1], answer))
