import os
import numpy as np
import random
from scipy.stats import entropy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import defaultdict
from scipy.spatial import distance
from matplotlib.patches import Circle
import pandas as pd
from tqdm import tqdm
import math
from scipy import stats
import time

threemonths = 3 * 60

# Define the Agent class
class Agent:
    def __init__(self, agent_id, wealth, personality):
        self.agent_id = agent_id
        self.wealth = wealth
        self.transaction_history = []
        self.wealth_spending = []
        self.positive_history = 0
        self.negative_history = 0
        self.personality = personality  # Add a personality attribute

    def update_wealth(self, amount):
        if self.personality == 'loss_aversion':
            self.wealth += amount * 0.9  # Loss averse agents save more
        elif self.personality == 'overconfidence':
            self.wealth += amount * 1.1  # Overconfident agents spend more
        elif self.personality == 'altruism':
            self.wealth += amount  # Altruistic agents donate some of their wealth
        elif self.personality == 'envy':
            if random.random() < 0.5:  # 50% chance to spend more
                self.wealth += amount * 1.2
            else:  # 50% chance to save more
                self.wealth += amount * 0.8
        else:
            self.wealth += amount  # Default behavior for agents with no specified personality

        self.wealth = max(0, self.wealth)  # Ensure wealth doesn't go negative

    def add_transaction(self, transaction):
        self.transaction_history.append(transaction)

    def update_history(self, purchaser_votes, seller_votes, average_votes, spent_votes):
        if spent_votes != 0:
            self.positive_history = average_votes * (purchaser_votes / spent_votes)
            self.negative_history = average_votes * (seller_votes / spent_votes)
        else:
            self.positive_history = 0
            self.negative_history = 0

# Define the Transaction class
class Transaction:
    def __init__(self, sender, receiver, amount, time):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.time = time

# Define the Economy class
class Economy:
    def __init__(self, agents, tax_scheme, start_point=None):  # Add start_point parameter
        self.agents = agents
        self.transactions = []
        self.tax_scheme = tax_scheme
        self.agent_counts_over_time = []
        self.final_wealth_distributions = []
        self.travel_distance = 0
        self.internal_points = 0
        self.treasury = 0
        self.current_point = None
        self.start_point = start_point  # Set the start_point attribute

    def get_wealth_distribution(self):
        return [agent.wealth for agent in self.agents]

    def initial_distribution(self):
        return np.array([agent.wealth for agent in self.agents])

    def perform_taxes(self):
        total_tax = 0
        for agent in self.agents:
            tax = self.tax_scheme(agent.wealth)
            tax = min(tax, agent.wealth)  # Prevent tax from exceeding agent's wealth
            agent.update_wealth(-tax)
            total_tax += tax
        self.treasury += total_tax

    def perform_nianomics(self):
        average_wealth = np.mean([agent.wealth for agent in self.agents])
        total_donation_rate = 0
        for agent in self.agents:
            agent.update_history(agent.wealth, agent.wealth, average_wealth, agent.wealth)
            # Calculate the amount to be donated or received in each tick
            wealth_adjustment = (average_wealth - agent.wealth) / (threemonths / 3)
            # Adjust the agent's wealth in each tick
            agent.update_wealth(wealth_adjustment)
            # Calculate the donation rate for agents with wealth above average
            if agent.wealth > average_wealth:
                donation_rate = (agent.wealth - average_wealth) / (threemonths / 3)
                agent.update_wealth(-donation_rate)
                total_donation_rate += donation_rate
        self.treasury -= total_donation_rate

    def balance_treasury(self):
        total_wealth = sum([agent.wealth for agent in self.agents])
        average_wealth = (total_wealth + self.treasury) / (len(self.agents) + 1)  # Add 1 to account for treasury
        if abs(self.treasury - average_wealth) > 1e-6:  # If the treasury and average wealth are not equal
            for agent in self.agents:
                agent.update_wealth(average_wealth - agent.wealth)
            self.treasury = average_wealth

# Define the tax algorithm
def tax_algorithm(params):
    base_rate, wealth_threshold, additional_rate = params
    def calculate_tax(wealth):
        if wealth > wealth_threshold:
            return base_rate * wealth + additional_rate * (wealth - wealth_threshold)
        else:
            return base_rate * wealth
    return calculate_tax

# Define the CustomOptimizer class
class CustomOptimizer:
    def __init__(self, bounds, start_point, step_size):
        self.bounds = bounds
        self.start_point = np.array(start_point)  # Convert to numpy array
        self.step_size = step_size
        self.memory = []
        self.unsuccessful_runs = 0
        self.consecutive_unsuccessful_runs = 0
        self.winners = []
        self.current_point = np.array(start_point)  # Convert to numpy array
        self.current_score = float("-inf")
        self.no_improvement_streak = 0
        self.iterations = 0
        self.max_consecutive_failures = 10

    def random_step(self):
        return np.random.uniform(-self.step_size, self.step_size, len(self.current_point))

    def save_winner(self, point, score, folder):
        os.makedirs(folder, exist_ok=True)
        np.savetxt(os.path.join(folder, 'winner_{}.txt'.format(len(self.winners))), point)
        with open(os.path.join(folder, 'winner_{}_details.txt'.format(len(self.winners))), 'w') as file:
            file.write('Score: {}\n'.format(score))
            file.write('Taxation Algorithm:\n')
            file.write(str(tax_algorithm(point)))


    def evaluate(self, point):
        return objective_function(point, self.memory)

    def explore(self):
        while self.consecutive_unsuccessful_runs < self.max_consecutive_failures:
            # Calculate forces
            force_vector = np.zeros_like(self.current_point)
            if self.no_improvement_streak > 0:
                direction_to_start = self.start_point - self.current_point
                force_vector += direction_to_start * (self.no_improvement_streak / 10.0)

            for winner in self.winners:
                direction_to_winner = np.array(winner[0]) - self.current_point  # Convert to numpy array
                norm = np.linalg.norm(direction_to_winner)
                if norm != 0 and not np.any(np.isnan(direction_to_winner)):
                    force_vector -= direction_to_winner / norm

            # Update current point with random step and forces
            step_vector = self.random_step() + force_vector
            lower_bounds, upper_bounds = np.array(self.bounds).T
            new_point = self.current_point + step_vector
            new_point = np.clip(new_point, lower_bounds, upper_bounds)
            new_score = self.evaluate(new_point)

            # Check if the new point is an improvement
            if new_score > self.current_score:
                self.current_point = new_point
                self.current_score = new_score
                self.winners.append((new_point, new_score))
                self.save_winner(new_point, new_score, 'winners')
                self.consecutive_unsuccessful_runs = 0
                self.no_improvement_streak = 0
            else:
                self.consecutive_unsuccessful_runs += 1
                self.no_improvement_streak += 1

            # Check if there's an improvement after a certain number of unsuccessful runs
            if self.consecutive_unsuccessful_runs >= 10:
                # Randomly select one of the previous winners
                if self.winners:
                    random_winner = random.choice(self.winners)
                    self.current_point, self.current_score = random_winner
                    self.consecutive_unsuccessful_runs = 0

            # Save the current point and score to memory
            self.memory.append((self.current_point, self.current_score))

            # Print real-time readouts every 10 iterations
            if self.iterations % 10 == 0:
                print("\nIteration:", self.iterations)
                print("Current Point:", self.current_point)
                print("Current Score:", self.current_score)

            # Increment the iteration count
            self.iterations += 1

# Define the objective function
def objective_function(point, memory):
    # Create an economy with the given tax algorithm
    num_agents = 11192
    total_wealth = 12390800
    initial_wealth_per_agent = total_wealth / num_agents
    personalities = ['loss_aversion', 'overconfidence', 'altruism', 'envy', 'default']
    agents = [Agent(i, initial_wealth_per_agent, random.choice(personalities)) for i in range(num_agents)]  # Create a list of 11192 agents with initial wealth of approximately $1107.19 and random personalities
    economy = Economy(agents, tax_algorithm(point))  # Create an economy with the agents and a tax algorithm

    # Run the economy for a certain amount of time
    start_time = time.time()
    while time.time() - start_time < threemonths:  # Run for X seconds times some number
        economy.perform_taxes()
        economy.perform_nianomics()
        economy.balance_treasury()

    # Calculate the score based on the final wealth distribution
    final_distribution = economy.get_wealth_distribution()
    score = entropy(final_distribution)

    # Add the economy to the memory
    memory.append((point, score, economy))

    return score

# Define the main function
if __name__ == "__main__":
    log = []
    bounds = [(0, 1), (0, 1), (0, 1)]
    start_point = np.random.uniform(0, 1, len(bounds))
    optimizer = CustomOptimizer(bounds, start_point, 0.1)

    # Start exploration
    optimizer.explore()
