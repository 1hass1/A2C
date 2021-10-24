# A2C
# Cutting Edge RL for Breakout Game


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:59:30 2020

@author: hassan
"""

import time
import joblib
import numpy as np
import tensorflow as tf
import os

# these things are for when we generate random #s, we want to be reproducible
def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)

# to calculate the entropy of the categorical distribution
# in this case we have the logits and not the probabilities
# so, we can do some tricks to make the calculation more precise
# in order to calculate the softmax, we 1st to exponentiate the logits, then divide by the sum of those exponentials
# the trick is, before exponentiating, we subtract the max value from all the elements
# this is bcs performing the exponent can lead to very large numbers which can lead to numerical overflow
# the 2nd trick, we dont need the probability p to calculate the log of p, if we have the logits
def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1)

# this function just grabs the tensorflow variables within a given scope
# normally, gradient descent is performed automatically by tensorflow, so we dont need a function like this
# but we will see why we need to grab the parameters of the NN later in code
def find_trainable_variables(key):
    with tf.compat.v1.variable_scope(key):
        return tf.compat.v1.trainable_variables()

# this function will calculate the discounted end step return, given a list of rewards and done flags
# noteably, the last element of the rewards list will be the value V(s) for the last state
# we just pass it in as 1 list since its not being treated differently from the other elements
# also, see how we use the done flag
""" (same as..
  if done:
    reward
  else:
    reward + gamma*V(s')
  )"""
# usually, our estimated return is r+gamma*V(s'), but it is only the case if s' is not a terminal state
# if s' is a terminal state, the value is always 0, bcs there cant be any future rewards
# so, when multiplying by (1 - done), we are saying V(s') = 0 (if next state is terminal)
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]   # ::-1 means reverse the list



class Agent:
    def __init__(self, Network, ob_space, ac_space, nenvs, nsteps, nstack,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6)):
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=nenvs,
                                inter_op_parallelism_threads=nenvs)
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        nbatch = nenvs * nsteps
        
        tf.compat.v1.disable_eager_execution()
        
        # each placeholder will be created with cst batch size (also what we did when we defined the NN)
        # this will be equal to the # of environments x # time steps b/w each update
        # this is bcs our algorithm is synchronous
        # so if we have 8 environments and 5 steps b/w each update, then a batch size = 8 x 5 = 40
        
        # A is the action we perform during each step, which is why it's an int32 and a 1D array of size nbatch
        A = tf.compat.v1.placeholder(tf.int32, [nbatch])
        # ADV is the advantage, it's an array of floats of size nbatch
        ADV = tf.compat.v1.placeholder(tf.float32, [nbatch])
        # R is the return
        # why do we need advantages and rewards??
        # bcs advantage is for training the policy while the return is the target for the value network
        R = tf.compat.v1.placeholder(tf.float32, [nbatch])
        # LR is the learning rate (can be removed since its just a cst)
        LR = tf.compat.v1.placeholder(tf.float32, [])

        # even though these are 2 different NNs, they refer to the same NN with the same weights
        # the difference is reuse = false and reuse = true
        # also, nsteps = 1, and nsteps = nsteps (which is 5)
        step_model = Network(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = Network(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)
        # this is bcs the batch size is cst. so when we are training,
        # we're collecting data from different environments and talking multiple steps so our batch size = 40 (if # parallel environments = 8)
        # but when we want to make a prediction for the next action, we should take in the current state which represent only 1 step
        # so, since the placeholders have cst sample size, we need to have 2 separate models with separate placeholders
        
        # next, we create our loss
        # in the NN file, we only defines the layers (architecture of the NN), not the objective or the algorithm used to train the model
        # remember, we have 2 heads for our NN: policy and V(s). each of these has a corresponding loss
        # details: bcs we have 2 things 2 optimize, there are multiple solutions
        # solution 1: do the updates in an alternating fashion. 1st, we do 1 step of GD w.r.t. policy loss, then 1 step of GD w.r.t. value loss. then again policy loss....
        # solution 2: to make it all 1 big loss, but the loss must be scalar so that we can optimize it
        # so, how to get scalar loss if we have 2 different outputs each having different objectives
        # the solution here is to just add them together
        # we could add the 2 losses directly (policy and value loss)
        # but what we did here is that we weighted these 2 losses relative to each other
        # we can see here, the policy gradient loss has a weight of 1, whereas the entropy has a coefficient of 0.01
        # and the value function loss has a coefficient of 0.5
        # (These are Hyperparameters that we can tune for experimentation)
        # the softmax function used below can be thought to be weird when using it here (since it's for classification)
        # but, recall what it actually calculates:
        # given some output probability Y, and a target T:
          # it calculates the sum (T x logY), where T and Y are vectors, if T is one-hot encoded (embedded)
          # but if T is not one-hot encoded (if it is an index from 0 to k-1),
          # then it calculates log Y(T), which returns the equivalent result
        # in our case, we want log pi(a), where a is the action
        # we know that a is represented by integers starting from 0, so it makes sense
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        # next step is to multiply it by the advantages, that gives us the policy gradient loss
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        # as usual, the loss for the value function is the squared error b/w value function prediction and R (the placeholder of the return)
        # also, the extra division by 2 is not normally done (example of hidden hyperparameters)
        # also, the coeff for each part of the loss
        vf_loss = tf.reduce_mean(tf.math.squared_difference(tf.squeeze(train_model.vf), R) / 2.0)
        # lastly, we calculate the entropy of the distribution, which also gets added to our loss
        # this to aid and exploration which can be though of as regularization (we use the cat_entropy function defined before)
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        
        # next, we define the optimizer, not the usual 1 line of code
        # 1st we get the parameters of the model using the function we defined before
        params = find_trainable_variables("model")
        # then, we calculate the gradients of these parameters
        grads = tf.gradients(loss, params)
        # then, we clip these gradients using a max norm of 0.5 (another hidden detail)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        # finally, we join these clips gradients with the parameters and pass these into the RMSPropOptimizer
        grads_and_params = list(zip(grads, params))
        trainer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads_and_params)
        
        
        # this function will run the training step we just defined
        # taken in a set of states, rewards, actions, and values
        def train(states, rewards, actions, values):
            # 1st, we calculate the advantages, which is rewards - values
            # here, rewards means the end step reward (but sometimes we mean the return)
            advs = rewards - values
            # next we create feed_dict for all the items we have to pass into the session
            # next, we use the session to run a training step and we also return each component of the loss (policy loss, value loss, and policy entropy)
            feed_dict = {train_model.X: states, A: actions, ADV: advs, R: rewards, LR: lr}
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                feed_dict
            )
            return policy_loss, value_loss, policy_entropy
        
        
        # the save and load functions can save and load the weights of the NN
        # this is another reason why we need a function to grab all the parameters of the model
        def save(save_path):
          # here, we grab the parameters using the session, and then dumping those parameters using joblib
          ps = sess.run(params)
          joblib.dump(ps, save_path)

        def load(load_path):
          # in the load function, we use tensorflow assign, to assign those parameters back to the actual variables in the NN
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)
        
        # lastly, we set some attributes for the agent class, and run run some tf boilerplates code to initialize the tf variables 
        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.save = save
        self.load = load
        tf.compat.v1.global_variables_initializer().run(session=sess)

# In this class, the agent is responsible for taking each step in the environment
# the runner is responsible for looping over the steps and accummulating th data, meaning the states, rewards,  actions, and done flags
class Runner:
  # in the constructor, we'ree basically setting up a bunch of instance variables (simple)
  # access the env, agent, dimensions of each frame, dimensions of each batch,...
    def __init__(self, env, agent, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.agent = agent
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        # interesting that self.state is a pre-initialized array
        # that will help with efficiency, since we dont need to keep re-allocating memory when we update the current state
        # (we can see that in the next function: update_state())
        # when we receive a new observation, we shift the existing observations and add the new observation at the end
        self.state = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.update_state(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        self.total_rewards = [] # store all workers' total rewards
        self.real_total_rewards = []

    def update_state(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
        self.state = np.roll(self.state, shift=-self.nc, axis=3)
        self.state[:, :, :, -self.nc:] = obs
    
    
    # the main function in this class
    def run(self):
      # 1st, we initialize empty lists, mb stands for mini batch
        mb_states, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        # next, we do a predefines # steps (5 in our case)
        for n in range(self.nsteps):
          # 1st, we call agent.step, meaning we give the agent a state, and ask it to sample an action and and return the predicted value for that state
            actions, values = self.agent.step(self.state)
            # next, we add all this data to our mini baches
            # then we perform the action (actually, a vector of actions (bcs we have a vector of environments)) we sampled in the environment
            mb_states.append(np.copy(self.state))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, infos = self.env.step(actions)
            # after performing the actions, we check the done flags and the info dictionaries, since these carry the total rewards we received 
            for done, info in zip(dones, infos):
                if done:
                  # we store the rewards for each episode in self.total_rewards
                  # and we store the rewards at the end of each game, meaning we've lost all five lives in self.real_total_rewards

                    self.total_rewards.append(info['reward'])
                    if info['total_reward'] != -1:
                        self.real_total_rewards.append(info['total_reward'])
            # after that, everything in this function is just organizing the return values
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.state[n] = self.state[n] * 0
            self.update_state(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_states = np.asarray(mb_states, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_dones = mb_dones[:, 1:]
        # calculating the values V(s) of the final state 
        last_values = self.agent.value(self.state).tolist()
        # discount/bootstrap off value fn
        # next, we calculate the n step discounted returns (we call it reward in the code)
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
              # here, we use V(s) in the discounted return calculations, only if the final state in not a terminal state
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        return mb_states, mb_rewards, mb_actions, mb_values


def learn(network, env, seed, new_session=True,  nsteps=5, nstack=4, total_timesteps=int(80e6),
          vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
          epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=1000):
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(seed)

    nenvs = env.num_envs
    env_id = env.env_id
    save_name = os.path.join('models', env_id + '.save')
    ob_space = env.observation_space
    ac_space = env.action_space
    agent = Agent(Network=network, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs,
                  nsteps=nsteps, nstack=nstack,
                  ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps)
    if os.path.exists(save_name):
        agent.load(save_name)

    runner = Runner(env, agent, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()
    for update in range(1, total_timesteps // nbatch + 1):
        states, rewards, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = agent.train(
            states, rewards, actions, values)
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            print(' - - - - - - - ')
            print("nupdates", update)
            print("total_timesteps", update * nbatch)
            print("fps", fps)
            print("policy_entropy", float(policy_entropy))
            print("value_loss", float(value_loss))

            # total reward
            r = runner.total_rewards[-100:] # get last 100
            tr = runner.real_total_rewards[-100:]
            if len(r) == 100:
                print("avg reward (last 100):", np.mean(r))
            if len(tr) == 100:
                print("avg total reward (last 100):", np.mean(tr))
                print("max (last 100):", np.max(tr))

            agent.save(save_name)

    env.close()
    agent.save(save_name)
