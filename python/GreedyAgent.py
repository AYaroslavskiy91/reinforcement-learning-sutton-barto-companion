import numpy as np
import random

class GreedyAgent:
    
    def __init__(self, initial_reward, num_arms, epsilon=0, reward_structure=0, step_size=0):
        '''
        Initiates an instance of the greedy agent algorithm. 
        
        Argument reward_structure determines how rewards are calculated. A definition of possibilities is provided:
        0: normally distributed rewards centered at stationary normally distributed means. 
        1: Equal initial rewards that each take random walks
        '''
        
        self.epsilon = epsilon
        self.last_action = 0
        self.q_values = [initial_reward] * num_arms
        self.last_reward = initial_reward
        self.arm_count = [0] * num_arms
        self.reward_structure = reward_structure
        self.step_size = step_size
        
        if reward_structure == 0:
            self.q_star_means = np.random.normal(0,1,num_arms)
            
        if reward_structure == 1:
            self.q_star_means = np.zeros(num_arms)
        
    def normally_distributed_rewards(self):
        
        return np.random.normal(self.q_star_means,1)
    
    def non_stationary_random_walk_rewards(self):
        
        return np.add(self.q_star_means, np.random.normal(0, .01, len(self.arm_count)))
        
    def argmax(self):
        
        ties = []

        for i in range(len(self.q_values)):
            if self.q_values[i]==max(self.q_values):
                ties.append(i) 
        return np.random.choice(ties)

    def agent_step(self):
        if self.step_size == 0:
            self.arm_count[self.last_action] += 1
            self.q_values[self.last_action] = (
                self.q_values[self.last_action] 
                + 1/self.arm_count[self.last_action]*(self.last_reward - self.q_values[self.last_action])
            )
        else: 
            self.q_values[self.last_action] = (
                self.q_values[self.last_action] 
                + self.step_size*(self.last_reward - self.q_values[self.last_action])
            )
        
        if np.random.uniform()<self.epsilon:
            current_action = np.random.choice(range(len(self.q_values)))
        else:
            current_action = self.argmax()
        self.last_action = current_action
                   
        if self.reward_structure == 0:
            current_reward = self.normally_distributed_rewards()[current_action]
            
        elif self.reward_structure == 1: 
            self.q_star_means = self.non_stationary_random_walk_rewards()
            current_reward = self.q_star_means[current_action]
        self.last_reward = current_reward
        
        return current_action, current_reward
    
    
    