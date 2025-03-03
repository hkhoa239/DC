import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # array of all nodes
        self.data = np.zeros(capacity, dtype=object) 
        self.data_pointer = 0 # current pointer to the data

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0 # replace when exceed the capacity

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change 

    def get_leaf(self,value):
        parent_index = 0
        while True:
            left_child = parent_index * 2 + 1
            right_child = left_child + 1

            if left_child >= len(self.tree):
                leaf_index = parent_index
                break

            if value <= self.tree[left_child]:
                parent_index = left_child
            else:
                value -= self.tree[left_child]
                parent_index = right_child

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    def total_priority(self):
        return self.tree[0]
    

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.alpha = 0.6 
        self.beta = 0.4
        self.epsilon = 1e-5
        self.beta_increment = 0.001 

    def __len__(self):
        return len(self.tree.data)

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, experience)

    def sample(self, batch_size):
        # get a minibatch from buffer 
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch_size

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value  = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            priorities.append(priority)
            batch.append(data)
            indices.append(index)

        # calculate Improtance Sampling weights
        sampling_probabilities = priorities / self.tree.total_priority()
        IS_weights = (len(self.tree.data) * sampling_probabilities) ** (-self.beta)
        IS_weights /= IS_weights.max() # normalize 

        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, indices, IS_weights
    
    def update_priority(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            updated_priority = np.abs(td_error) + self.epsilon 
            self.tree.update(i, updated_priority ** self.alpha) 