import numpy as np

# Load the training data
data = open('kafka.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d chars, %d unique' % (data_size, vocab_size))

# Encode/Decode char/vector
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = { i:ch for i, ch in enumerate(chars)}
print(chars)
print (char_to_ix)
print(ix_to_char)

vector_for_char_a = np.zeros((vocab_size, 1))
vector_for_char_a[char_to_ix['a']] = 1
print(vector_for_char_a.ravel())

# Definition of the network
#hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1
#model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 #input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01 #hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01 #hidden to output
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))