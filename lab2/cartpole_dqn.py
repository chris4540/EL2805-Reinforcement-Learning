import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from utils.exp_folder import make_exp_folder
from utils.hparams import HyperParams

EPISODES = 1000 # Maximum number of episodes


#DQN Agent for the Cartpole
#Q function approximation with NN, experience replay, and target network
class DQNAgent:
    #Constructor for the agent (invoked when DQN is first called in main)
    def __init__(self, state_size, action_size, exp_folder, **kwargs):
        # If True, stop if you satisfy solution confition
        self.check_solve = False
        # If you want to see Cartpole learning, then change to True
        self.render = False

        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.exp_folder = make_exp_folder(exp_folder)

       # Modify here
        hparams = HyperParams(**kwargs)
        hparams.display()
        hparams.save_to_txt(self.exp_folder / "hparams.txt")
        hparams.save_to_json(self.exp_folder / "hparams.json")

        # Set hyper parameters for the DQN. Do not adjust those labeled as Fixed.
        self.discount_factor = hparams.discount_factor
        self.learning_rate = hparams.learning_rate
        self.target_update_frequency = hparams.target_update_frequency
        self.memory_size = hparams.memory_size
        # -----------------------------
        self.epsilon = 0.02 # Fixed
        self.batch_size = 32 # Fixed
        self.train_start = 1000 # Fixed

        # Number of test states for Q value plots
        self.test_state_no = 10000

        #Create memory buffer using deque
        self.memory = deque(maxlen=self.memory_size)

        #Create main network and target network (using build_model defined below)
        self.model = self.build_model()
        self.target_model = self.build_model()

        # save down the model summary
        with open(self.exp_folder / 'model_summary.txt','w') as fh:
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

        #Initialize target network
        self.update_target_model()

    def build_model(self):
        """
        Approximate Q function using Neural Network
        State is the input and the Q Values are the output.

        See also:
        https://keras.io/getting-started/sequential-model-guide/
        """
        # Edit the Neural Network model here
        model = Sequential()
        model.add(Dense(8, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # After some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """ Get action from model using epsilon-greedy policy

        Args:
            state ([type]): [description]

        Returns:
            [type]: An action
        """
        if np.random.binomial(1, self.epsilon) == 1:
            # random policy
            action = random.randrange(self.action_size)
        else:
            # e-greedy policy
            q_values = self.model.predict(state)
            action = q_values.argmax()
        return action
###############################################################################
###############################################################################
    #Save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #Add sample to the end of the list

    #Sample <s,a,r,s'> from replay memory
    def train_model(self):
        # Do not train if not enough memory
        if len(self.memory) < self.train_start:
            return

        # Train on at most as many samples as you have in memory
        batch_size = min(self.batch_size, len(self.memory))
        # Uniformly sample the memory buffer
        mini_batch = random.sample(self.memory, batch_size)
        # -----------------------------------------------------
        # Preallocate network and target network input matrices.
        # -----------------------------------------------------
        # batch_size by state_size two-dimensional array (not matrix!)
        update_input = np.zeros((batch_size, self.state_size))
        # Same as above, but used for the target network
        update_target = np.zeros((batch_size, self.state_size))

        # Empty arrays that will grow dynamically
        # action, reward, done = [], [], []
        action = list()
        reward = list()
        done = list()

        # for i in range(self.batch_size):
        for i in range(batch_size):
            # Allocate s(i) to the network input array from iteration i in the batch
            update_input[i] = mini_batch[i][0]
            # Store a(i)
            action.append(mini_batch[i][1])
            # Store r(i)
            reward.append(mini_batch[i][2])
            # Allocate s'(i) for the target network array from iteration i in the batch
            update_target[i] = mini_batch[i][3]
            # Store done(i)
            done.append(mini_batch[i][4])

        # Generate target values for training the inner loop network using the network model
        target = self.model.predict(update_input)
        # Generate the target values for training the outer loop target network
        target_val = self.target_model.predict(update_target)

        # -----------------------------------------------------------
        # Q Learning: get maximum Q value at s' from target network
        # Read Part7.pdf, page 29
        # -----------------------------------------------------------
        # for i in range(self.batch_size): #For every batch
        for i in range(batch_size):
            if done[i]:
                # if this is the episode ends
                target[i][action[i]] = reward[i]
            else:
                # Consider also the future reward (Q-value predicted by outer loop)
                target[i][action[i]] = reward[i] + self.discount_factor * np.max(target_val[i])

        # Train the inner loop network
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
        return
    #Plots the score per episode as well as the maximum q value per episode, averaged over precollected states.
    def plot_data(self, episodes, scores, max_q_mean):
        pylab.figure(0)
        pylab.plot(episodes, max_q_mean, 'b')
        pylab.xlabel("Episodes")
        pylab.ylabel("Average Q Value")
        pylab.savefig(self.exp_folder / "qvalues.png")

        pylab.figure(1)
        pylab.plot(episodes, scores, 'b')
        pylab.xlabel("Episodes")
        pylab.ylabel("Score")
        pylab.savefig(self.exp_folder / "scores.png")

###############################################################################
###############################################################################

if __name__ == "__main__":
    # change the exp folder here
    # exp_folder = "experiments/nn_size"
    discount_factor = 0.9
    exp_folder = "experiments/discount_factor_{}".format(str(discount_factor).replace(".", ""))

    #For CartPole-v0, maximum episode length is 200
    env = gym.make('CartPole-v0') #Generate Cartpole-v0 environment object from the gym library
    #Get state and action sizes from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    print("state_size: ", state_size)
    print("action_size: ", action_size)

    # Create agent, see the DQNAgent __init__ method for details
    agent = DQNAgent(state_size, action_size, exp_folder=exp_folder, discount_factor=discount_factor)

    # Collect test states for plotting Q values using uniform random policy
    test_states = np.zeros((agent.test_state_no, state_size))
    max_q = np.zeros((EPISODES, agent.test_state_no))
    max_q_mean = np.zeros((EPISODES,1))

    done = True
    for i in range(agent.test_state_no):
        if done:
            done = False
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            test_states[i] = state
        else:
            action = random.randrange(action_size)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            test_states[i] = state
            state = next_state

    scores, episodes = [], [] #Create dynamically growing score and episode counters
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset() #Initialize/reset the environment
        #Reshape state so that to a 1 by state_size two-dimensional array
        # i.e. [x_1,x_2] to [[x_1,x_2]]
        state = np.reshape(state, [1, state_size])
        # Compute Q values for plotting
        tmp = agent.model.predict(test_states)
        max_q[e][:] = np.max(tmp, axis=1)
        max_q_mean[e] = np.mean(max_q[e][:])

        while not done:
            if agent.render:
                env.render() #Show cartpole animation

            #Get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size]) #Reshape next_state similarly to state

            #Save sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            #Training step
            agent.train_model()
            score += reward #Store episodic reward
            state = next_state #Propagate state

            if done:
                #At the end of very episode, update the target network
                if e % agent.target_update_frequency == 0:
                    agent.update_target_model()
                #Plot the play time for every episode
                scores.append(score)
                episodes.append(e)

                print("episode:", e, "  score:", score," q_value:", max_q_mean[e],"  memory length:",
                      len(agent.memory))

                # if the mean of scores of last 100 episodes is bigger than 195
                # stop training
                if agent.check_solve:
                    if np.mean(scores[-min(100, len(scores)):]) >= 195:
                        print("solved after", e-100, "episodes")
                        agent.plot_data(episodes,scores,max_q_mean[:e+1])
                        sys.exit()
    agent.plot_data(episodes,scores,max_q_mean)
