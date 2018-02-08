
# coding: utf-8

# In[1]:


import os.path
import gzip
import pickle
import os
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath("MNIST_data"))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    if os.path.exists(file_path):
        return
    print("Downloading" + file_name + " ... ")
    
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

def download_mnist():
    for v in key_file.values():
        _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    print("converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    print("Converting " + file_name + " to Numpy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, img_size)
    print("Done")
    return data

def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    return dataset

def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")
    

def load_mnist(normalize=True, flatten=True):
    if not os.path.exists(save_file):
        init_mnist()
    
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
   
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


# In[2]:


(x_train, y_train), (x_test, y_test) = load_mnist()


# In[3]:


def enc_one_hot(y, num_labels=10):
    one_hot = np.zeros((num_labels, y.shape[0]))
    for i, val in enumerate(y):
        one_hot[val, i] = 1.0
    return one_hot

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(X):
    return (1.0 - sigmoid(X)) * sigmoid(X)

def softmax(z):
    logC = -np.max(z)
    return np.exp(z + logC)/np.sum(np.exp(z + logC), axis = 0)

def cross_entropy_error(y, t):
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def add_bias_unit(X, where):
    if where == 'column':
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
    elif where == 'row':
        X_new = np.ones((X.shape[0] + 1, X.shape[1]))
        X_new[1:, :] = X

    return X_new

def initialize_weights(n_features, n_hidden, n_output):
    w1 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_features+1) )
    w1 = w1.reshape(n_hidden, n_features+1)
    w2 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_hidden+1) )
    w2 = w2.reshape(n_hidden, n_hidden + 1)
    w3 = np.random.uniform(-1.0, 1.0, size=n_output*(n_hidden+1) )
    w3 = w3.reshape(n_output, n_hidden+1)
    return w1, w2, w3


# In[4]:


def compute_dropout(activations, dropout_prob = 0.5):
    activations/=dropout_prob    
    mult = np.random.binomial(1, 0.5, size = activations.shape)
    activations*=mult
    return activations


# In[5]:


def feed_forward(X, w1, w2, w3, do_dropout = True):
    a1 = add_bias_unit(X, where = 'column')
    if do_dropout: a1 = compute_dropout(a1)
    z2 = w1.dot(a1.T)
    a2 = sigmoid(z2)
    
    a2 = add_bias_unit(a2, where = 'row')
    if do_dropout: a2 = compute_dropout(a2)
    z3 = w2.dot(a2)
    a3 = sigmoid(z3)
    
    a3 = add_bias_unit(a3, where = 'row')
    if do_dropout: a3 = compute_dropout(a3)
    z4 = w3.dot(a3)
    a4 = softmax(z4)
    
    return a1, z2, a2, z3, a3, z4, a4


# In[6]:


def predict(X, w1, w2, w3):
    a1, z2, a2, z3, a3, z4, a4 = feed_forward(X, w1, w2, w3, do_dropout = False)
    y_pred = np.argmax(a4, axis = 0)
    return y_pred


# In[7]:


def calc_grad(a1, a2, a3, a4, z2, z3, z4, y_enc, w2, w3):
    delta4 = a4 - y_enc
    z3 = add_bias_unit(z3, where = 'row')
    delta3 = w3.T.dot(delta4)*sigmoid_grad(z3)
    delta3 = delta3[1:, :]
    z2 = add_bias_unit(z2, where = 'row')
    delta2 = w2.T.dot(delta3)*sigmoid_grad(z2)
    delta2 = delta2[1:, :]
    
    grad1 = delta2.dot(a1)
    grad2 = delta3.dot(a2.T)
    grad3 = delta4.dot(a3.T)
    
    return grad1, grad2, grad3


# In[8]:


def run_model(X, y, X_t, y_t):
    X_copy, y_copy = X.copy(), y.copy()
    y_enc = enc_one_hot(y)
    epochs = 400
    batch = 50
    
    w1, w2, w3 = initialize_weights(784, 75, 10)
        
    alpha = 0.9
    eta = 0.001
    dec = 0.00001
    delta_w1_prev = np.zeros(w1.shape)
    delta_w2_prev = np.zeros(w2.shape)
    delta_w3_prev = np.zeros(w3.shape)
    total_cost = []
    previous_accuracies = []
    pred_acc = np.zeros(epochs)
    
    for i in range(epochs):
        
        shuffle = np.random.permutation(y_copy.shape[0])
        X_copy, y_enc = X_copy[shuffle], y_enc[:, shuffle]
        eta /= (1+dec*i)
        
        mini = np.array_split(range(y_copy.shape[0]), batch)
        
        for step in mini:
            a1, z2, a2, z3, a3, z4, a4 = feed_forward(X_copy[step], w1, w2, w3)
            
            cost = softmax_loss(y_enc[:, step], a4)
            total_cost.append(cost)
            
            grad1, grad2, grad3 = calc_grad(a1, a2, a3, a4, z2, z3, z4, y_enc[:,step], w2, w3)
            delta_w1,delta_w2,delta_w3 = eta * grad1, eta * grad2, eta * grad3            
            w1 -= delta_w1 + alpha * delta_w1_prev
            w2 -= delta_w2 + alpha * delta_w2_prev
            w3 -= delta_w3 + alpha * delta_w3_prev
            
            delta_w1_prev, delta_w2_prev, delta_w3_prev = delta_w1, delta_w2, delta_w3
            
    
                
        y_pred = predict(X_t, w1, w2, w3)
        pred_acc[i] = 100*np.sum(y_t == y_pred, axis = 0) / X_t.shape[0]
        print('epoch #', i, 'acc #', pred_acc[i] )
    return total_cost, pred_acc


# In[9]:


cost, acc = run_model(x_train, y_train, x_test, y_test) 

