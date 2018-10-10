import numpy as np

input_data = np.array([2,3])

weights = {
        'node_0': np.array([1,1]),
        'node_1': np.array([-1,1]),
        'output': np.array([2,-1])
        }

node_0_val = np.dot(input_data,weights['node_0'])
node_1_val = np.dot(input_data,weights['node_1'])
node_2_val = np.dot(np.array([node_0_val,node_1_val]),weights['node_2'])

print("Result = {}".format(node_2_val))
