'''
20181015
'''
import os
import numpy as np
from softArm_env import softArm
import tensorflow as tf
import matplotlib.pyplot as plt
import queue
class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.0001,
            reward_decay=0.7,
            e_greedy=1,
            replace_target_iter=2000,
            memory_size=20000,
            batch_size=256,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0.9 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized    # decide to use double q or not

        self.learn_step_counter = 0

        self.model_dir = os.path.join(os.getcwd(), "modelofddqn")
        self.model_name = "{}.ckpt".format('DDQN_PRIOR')
        

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)] #assign -> 将t变为e

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
    
    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer, trainable):
            s_flat = tf.reshape(s, [-1, 4])
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(s_flat, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l1], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l1, n_l1], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b3 = tf.get_variable('b3', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [n_l1, n_l1], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b4 = tf.get_variable('b4', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
            
            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [n_l1, n_l1], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b5 = tf.get_variable('b5', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l5 = tf.nn.relu(tf.matmul(l4, w5) + b5)
            
            with tf.variable_scope('l6'):
                w6 = tf.get_variable('w6', [n_l1, n_l1], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b6 = tf.get_variable('b6', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l6 = tf.nn.relu(tf.matmul(l5, w6) + b6)

            with tf.variable_scope('l7'):
                w7 = tf.get_variable('w7', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b7 = tf.get_variable('b7', [1, self.n_actions], initializer=b_initializer, collections=c_names,  trainable=trainable)
                out = tf.matmul(l6, w7) + b7
            
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None,self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 50, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer, True)

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None,self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, False)
        self.saver = tf.train.Saver()
        

    def store_transition(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            # print(s,a,r,s_)
            transition = np.hstack((s, np.array([a, r]), s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
            q_value = actions_value[0][action]
            
        else:
            action = np.random.randint(0, self.n_actions)
            q_value = 0
        return action ,q_value

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: batch_memory[:, -self.n_features:],
                           self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return self.cost

    def load_model(self):
        # load from model_path
        model_path = os.path.join(self.model_dir, self.model_name)
        self.saver.restore(self.sess, model_path)
        # else:
        #     # load from checkpoint
        #     checkpoint = tf.train.get_checkpoint_state(self.model_dir)
        #     if checkpoint and checkpoint.model_checkpoint_path:
        #         self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def save_model(self):
        self.saver.save(self.sess, os.path.join(self.model_dir, self.model_name))
        print('MODEL saved')
    def set_start_epsilon(self,new_epsilon):
        self.epsilon = new_epsilon

env = softArm()
MODE = False #False = train 
MODE = True #,True = test
with tf.variable_scope('DQN_with_prioritized_replay'):
    if MODE == False:
        RL_prio = DQNPrioritizedReplay(
            n_actions=4, n_features=4, memory_size=10000,
            e_greedy_increment=0.001, prioritized=True, output_graph=False,
        )
    else:
        RL_prio = DQNPrioritizedReplay(
            n_actions=4, n_features=4, memory_size=10000,
            e_greedy_increment=0.01, prioritized=True, output_graph=False,
        )
try:
    RL_prio.load_model()
    print('MODEL loaded')
except Exception:
    print('MODEL not found')

def train(RL):
    MAX_STEP = 1000
    LRARN_START = 100
    EPSILON = 0.4
    EPSILON_INCREMENT = (0.75-EPSILON)/MAX_STEP
    win = 0
    # display_step = MAX_STEP-10
    win_rate_ = np.zeros([MAX_STEP-100,1])
    net_loss_list = np.zeros([MAX_STEP-100,1])
    q_value_list = np.zeros([MAX_STEP-100,1])
    win_rate_queue = queue.Queue(maxsize=101)
    i_episode = 0
    net_loss = 0
    q_value = 0
    current_epoch = 0
    while i_episode < MAX_STEP:
        current_epoch = 0 #记录小循环运行次数的mark
        net_loss = 0
        q_value = 0
        observation, reward, terminal = env.observe()
        while not terminal:
            action, current_q_value= RL.choose_action(observation)
            q_value += current_q_value
            env.execute_action(action)
            observation_, reward, terminal = env.observe()
            RL.store_transition(observation, action, reward, observation_)
            current_epoch += 1
            observation = observation_
            # if i_episode >= display_step:
            #     env.draw()
            if i_episode > LRARN_START:
                    net_loss += RL.learn()
        if reward == 2:
            win += 1
            win_rate_queue.put(1,block=False)
        else:
            win_rate_queue.put(0,block=False)
        if i_episode >= 100:
            win -= win_rate_queue.get(block=False)
            win_rate_[i_episode-100,0] = win
            net_loss_list[i_episode-100,0] = net_loss/current_epoch
            q_value_list[i_episode-100,0] = q_value/current_epoch
            print("WIN: {:03d}/{:03d} ({:.3f}%)".format(win, i_episode, win)+' ---LOADING: %.3f' % (i_episode/MAX_STEP))
        else:
            print(str(i_episode))
        env.reset()
        EPSILON += EPSILON_INCREMENT
        RL.set_start_epsilon(EPSILON)
        i_episode += 1
    
    plt.figure(2)
    plt.plot(win_rate_, c='r')

    plt.ylabel('win_rate_')
    plt.xlabel('episode')
    plt.grid()
    plt.show()
    
    with open('win_rate0411.txt','a') as file:
        for i in win_rate_:
            file.write(str(i[0]))
            file.write('\n')
    plt.figure(3)
    plt.plot(net_loss_list, c='r')

    plt.ylabel('net_loss')
    plt.xlabel('episode')
    plt.grid()
    plt.show()
    with open('net_loss0411.txt','a') as file:
        for ii in net_loss_list:
            file.write(str(ii[0]))
            file.write('\n')
    plt.figure(4)
    plt.plot(q_value_list, c='r')

    plt.ylabel('q_value')
    plt.xlabel('episode')
    plt.grid()
    plt.show()
    with open('q_value0411.txt','a') as file:
        for iii in q_value_list:
            file.write(str(iii[0]))
            file.write('\n')

def test(RL):
    MAX_STEP = 100
    win = 0
    win_rate_queue = queue.Queue(maxsize=21)
    win_rate_ = np.zeros([MAX_STEP-20,1])
    for i_episode in range(MAX_STEP):
        observation, reward, terminal = env.observe()
        while not terminal:
            action,current_q_value = RL.choose_action(observation)
            env.execute_action(action)
            observation_, reward, terminal = env.observe()
            observation = observation_

            # if i_episode >= MAX_STEP-20:
                # env.draw()
                # pass
        if reward == 2:
            win += 1
            win_rate_queue.put(1,block=False)
        else:
            win_rate_queue.put(0,block=False)
        if i_episode >= 20:
            win -= win_rate_queue.get(block=False)
            win_rate_[i_episode-20,0] = win
            print("WIN: {:03d}/{:03d} ({:.3f}%)".format(win, i_episode, win*5)+' ---LOADING: %.3f' % (i_episode/MAX_STEP))
        else:
            print(str(i_episode))
        env.reset()
    
    plt.figure(2)
    plt.plot(win_rate_, c='r', label='DQN with prioritized replay')
    plt.ylabel('win_rate_')
    plt.xlabel('episode')
    plt.grid()
    plt.show()


# his_natural = train(RL_natural)
if MODE == False:
    train(RL_prio)
    RL_prio.save_model()
else:
    test(RL_prio)
# compare based on first success
# plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')


# with tf.variable_scope('natural_DQN'):
#     RL_natural = DQNPrioritizedReplay(
#         n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
#         e_greedy_increment=0.00005, sess=sess, prioritized=False,
#     )