def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = np.dot(input_data,weights['node_0_0'])#(____ * ____).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = np.dot(input_data,weights['node_0_1'])
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = np.dot(hidden_0_outputs,weights['node_1_0'])
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = np.dot(hidden_0_outputs,weights['node_1_1'])
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    #print(hidden_1_outputs)
    model_output = np.dot(hidden_1_outputs,weights['output'])
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)

