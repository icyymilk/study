import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
x=np.array([[1,1]])
y=np.array([[0]])
W = [
    np.array([[-0.0053,0.3793],[-0.5820,-0.5204],[-0.2723,0.1896]],dtype=np.float32).T,
    np.array([-0.0140,0.5607,-0.0628],dtype=np.float32),
    np.array([[0.1528,-0.1745,-0.1135]],dtype=np.float32).T,
    np.array([-0.5516],dtype=np.float32)
]
def feed_forward(inputs,outputs,weights):
    pre_hidden = np.dot(inputs,weights[0])+weights[1]
    hidden = 1/(1+np.exp(-pre_hidden))
    pred_out = np.dot(hidden,weights[2])+weights[3]
    loss = np.mean(np.square(pred_out-outputs))
    return pre_hidden,hidden,loss

def update_weight(inputs,outputs,weights,lr):
    origin_weights = deepcopy(weights)
    temp_weights = deepcopy(weights)
    updated_weights = deepcopy(weights)
    original_loss = feed_forward(inputs,outputs,origin_weights)
    for i,layer in enumerate(origin_weights):
        for index,weight in np.ndenumerate(layer):
            temp_weights =deepcopy(weights)
            temp_weights[i][index]+=0.0001
            _loss_plus=feed_forward(inputs,outputs,temp_weights)
            grad = (_loss_plus-original_loss)/0.0001
            updated_weights[i][index] -=grad*lr
    return updated_weights,original_loss

def backpropagation(inputs,outputs,weights,lr):
    pre_hidden,hidden,pred_out,loss=feed_forward(inputs,weights)

    d_loss_d_pred=-2*(pred_out-outputs)
    d_W2 =np.dot(hidden.T,d_loss_d_pred)
    d_b3=np.sum(d_loss_d_pred,axis=0)

    d_loss_d_hidden = np.dot(d_loss_d_pred,weights[2].T)
    sigmoid_grad = hidden*(1-hidden)
    d_loss_d_pre_hidden = d_loss_d_hidden*sigmoid_grad
    d_W0=np.dot(inputs.T,d_loss_d_pre_hidden)
    d_b1= np.sum(d_loss_d_pre_hidden,axis=0)
    update_weight=deepcopy(weights)
    update_weight[0]-=lr*d_W0;
    update_weight[1] -= lr * d_b1;
    update_weight[2] -= lr * d_W2;
    update_weight[3] -= lr * d_b3;


losses = []
for epoch in range(100):
    W, loss = update_weight(x,y,W,0.01)
    losses.append(loss)
plt.plot(losses)
plt.title('loss over increasing number of epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.show()

pre_hidden=np.dot(x,W[0])+W[1]
hidden = 1/(1+np.exp(-pre_hidden))
pred_out =np.dot(hidden,W[2])+W[3]
print(pred_out)
