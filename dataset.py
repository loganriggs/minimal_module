import numpy as np

range_numbers = 200

#Initialize values
data = np.stack([[x,y,0,x+y,x*y] for x in range(range_numbers) for y in range(range_numbers)])

#Split into inputs and labels
add_input = data[:,0:3]
add_label = data[:,3:4]
mult_input = data[:,0:3]
mult_label = data[:,4:5]

'''
Algorithm:
1. train normally
2. Add regularizer/noise up to epsilon performance drop (can combine w/ 1)
3. Restrict dataset to [add/mult/ etc], and compare regularizer loss across mask of each neuron (w/ minimal epsilon perf drop)
4 (optional) same but w/ probabilistic mask
'''

#Combine add and mult
add_mult = np.concatenate((add_input, mult_input), axis=1)
add_mult_labels = np.concatenate((add_label, mult_label), axis=1)

print("Add mult shape: (" + str(len(data)) + ", 6) == ", add_mult.shape)
print("Add mult labels shape: ", add_mult_labels.shape)

#Combine mult and add
mult_add = np.concatenate((mult_input, add_input), axis=1)
mult_add_labels = np.concatenate((mult_label, add_label), axis=1)

#combine both
add_mult_both = np.concatenate((add_mult, mult_add), axis=0)
add_mult_both_labels = np.concatenate((add_mult_labels, mult_add_labels), axis=0)

#Create just add and Mult
add_mult_single = np.concatenate((add_mult[:,:3], add_mult[:,3:]), axis=0)
add_mult_single_labels = np.concatenate((add_mult_labels[:,0:1], add_mult_labels[:,1:2]), axis=0)


np.save("add_mult_data", add_mult_both)
np.save("add_mult_labels", add_mult_both_labels)

np.save("add_mult_single_data", add_mult_single)
np.save("add_mult_single_labels", add_mult_single_labels)
