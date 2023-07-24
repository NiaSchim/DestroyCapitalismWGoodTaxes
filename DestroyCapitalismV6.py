good, update this so that unfortunates can recieve debt between 20% to 100% the mode ammount of starting wealth (so the average ammount of wealth unless rich dorks are present then it is the ammount that most agents have which will be lower than the typical average). as for the percentage allotted to all rich dorks, whilst it will be random, the ammount taken from the rest of the population to create the random distribution of wealth among rich dorks ought to be between 50 and 90 % of the total wealth of the population when dealing with rich dorks. this is to see how this system operates to create equality in the face of extreme disparity.

import os
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import defaultdict
from scipy.spatial import distance
from matplotlib.patches import Circle
import random
import pandas as pd
from tqdm import tqdm
import math

class CustomOptimizer:
    def __init__(self, bounds, start_point, step_size):
        self.bounds = bounds
        self.start_point = start_point
        self.step_size = step_size
        self.memory = []
        self.unsuccessful_runs = 0
        self.winners = []
        self.current_point = start_point
        self.current_score = float('-inf')
        self.no_improvement_streak = 0

    def random_step(self):
        return np.random.uniform(-self.step_size, self.step_size, len(self.current_point))

    def save_winner(self, point, score, folder):
        np.savetxt(os.path.join(folder, f'winner_{len(self.winners)}.txt'), point)
        with open(os.path.join(folder, f'winner_{len(self.winners)}_details.txt'), 'w') as file:
            file.write(f'Score: {score}\n')
            file.write('Taxation Algorithm:\n')
            file.write(str(self.tax_algorithm(point)))

    def explore(self):
        force_vector = np.zeros_like(self.current_point)
        if self.no_improvement_streak > 0:
            direction_to_start = self.start_point - self.current_point
            force_vector += direction_to_start * (self.no_improvement_streak / 10.0)

        for winner in self.winners:
            direction_to_winner = winner - self.current_point
            force_vector -= direction_to_winner / np.linalg.norm(direction_to_winner)

        step_vector = self.random_step() + force_vector
        new_point = self.current_point + step_vector
        new_point = np.clip(new_point, self.bounds[0], self.bounds[1])
        new_score = self.evaluate(new_point)

        if new_score > self.current_score:
            self.current_score = new_score
            self.current_point = new_point
            self.no_improvement_streak = 0
            if len(self.winners) == 0 or new_score > self.winners[0][1]:
                self.winners = [(new_point, new_score)]
                self.save_winner(new_point, new_score, 'best-winners')
            elif new_score == self.winners[0][1]:
                self.winners.append((new_point, new_score))
            self.save_winner(new_point, new_score, 'winners')
        else:
            self.no_improvement_streak += 1
            if np.array_equal(self.current_point, self.start_point):
                self.current_point = np.random.uniform(self.bounds[0], self.bounds[1], len(self.current_point))
                self.unsuccessful_runs += 1
        self.memory.append((new_point, new_score))

class Agent:
    def __init__(self, agent_id, wealth):
        self.agent_id = agent_id
        self.wealth = wealth
        self.transaction_history = []

    def update_wealth(self, amount):
        self.wealth += amount

    def add_transaction(self, transaction):
        self.transaction_history.append(transaction)

class Transaction:
    def __init__(self, sender, receiver, amount, time):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.time = time

class Economy:
    def __init__(self, agents, transactions):
        self.agents = agents
        self.transactions = transactions

    def get_wealth_distribution(self):
        return [agent.wealth for agent in self.agents]

class TaxScheme:
    def __init__(self, base_rate, wealth_threshold, additional_rate):
        self.base_rate = base_rate
        self.wealth_threshold = wealth_threshold
        self.additional_rate = additional_rate

    def compute_tax(self, agent, economy):
        if agent.wealth > self.wealth_threshold:
            return self.base_rate * agent.wealth + self.additional_rate * (agent.wealth - self.wealth_threshold)
        else:
            return self.base_rate * agent.wealth

def tax_algorithm(params):
    def calculate_tax(wealth):
        tax = wealth * params[0]
        return tax
    return calculate_tax

def objective_function(params, log):
    total_score = 0
    for _ in range(5):
        economy = Economy(100, tax_algorithm(params))
        total_score += -simulate_economy(economy, iterations=1000, transactions_per_iteration=100, num_richdorks=0, num_unfortunates=0)
        economy = Economy(100, tax_algorithm(params))
        total_score += -simulate_economy(economy, iterations=1000, transactions_per_iteration=100, num_richdorks=10, num_unfortunates=0)
        economy = Economy(100, tax_algorithm(params))
        total_score += -simulate_economy(economy, iterations=1000, transactions_per_iteration=100, num_richdorks=0, num_unfortunates=10)
        economy = Economy(100, tax_algorithm(params))
        total_score += -simulate_economy(economy, iterations=1000, transactions_per_iteration=100, num_richdorks=10, num_unfortunates=10)
    average_score = total_score / 20
    log.append(params)
    return average_score

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def draw_progress_circles(supra_volume, travel_distance, internal_points):
    fig, ax = plt.subplots()

    # Normalizing progress
    progress_distance = travel_distance % supra_volume / supra_volume
    progress_points = internal_points % supra_volume / supra_volume

    # Calculate effort
    spread_magnitude = math.floor(travel_distance / supra_volume)
    experience_magnitude = math.floor(internal_points / supra_volume)
    progress_effort = experience_magnitude / (spread_magnitude + 0.0001) # Added a small constant to avoid division by zero

    # Creating the progress circles
    circle_distance = Circle((0.5, 0.5), progress_distance/2, edgecolor='black', facecolor='red', lw=2)
    circle_points = Circle((0.5, 0.5), progress_points/2, edgecolor='black', facecolor='green', lw=2, alpha=0.25)
    circle_effort = Circle((0.5, 0.5), progress_effort/2, edgecolor='black', facecolor='blue', lw=2)

    # Adding the circles to the axes
    ax.add_patch(circle_distance)
    ax.add_patch(circle_points)
    ax.add_patch(circle_effort)

    plt.show()

def simulate_economy(economy, tax_scheme, iterations=1000, transactions_per_iteration=100, num_runs=20, num_richdorks=10, num_unfortunates=10):
    wealth_over_time = []
    plt.ion()
    fig, ax = plt.subplots(3, 1, figsize=(15, 20))
    ax1, ax2, ax3 = ax.flatten()
    winners = []
    unsuccessful_runs = []
    spending_averages = []
    earning_averages = []
    agent_counts = defaultdict(int)
    wealth_spending = defaultdict(list)
    wealth_spending_counts = defaultdict(int)

    # Select some agents to be 'richdorks' and 'unfortunates'
    richdorks = random.sample(economy.agents, num_richdorks)
    unfortunates = random.sample([agent for agent in economy.agents if agent not in richdorks], num_unfortunates)

    # Distribute wealth among agents
    total_wealth = sum(agent.wealth for agent in economy.agents)
    mode_wealth = stats.mode([agent.wealth for agent in economy.agents])[0][0]
    wealth_for_unfortunates = total_wealth * random.uniform(0.5, 0.9)  # 50% to 90% of total wealth

    for agent in unfortunates:
        debt = mode_wealth * random.uniform(0.2, 1.0)  # Debt is 20% to 100% of mode wealth
        agent.update_wealth(-debt)
        wealth_for_unfortunates += debt

    for agent in richdorks:
        extra_wealth = wealth_for_unfortunates / num_richdorks
        agent.update_wealth(extra_wealth)

    for i in tqdm(range(iterations)):
        for _ in range(transactions_per_iteration):
            sender = random.choice(economy.agents)
            while sender.wealth <= 0:  # Ensure sender has some wealth
                sender = random.choice(economy.agents)
            receiver = random.choice(economy.agents)
            amount = sender.wealth * random.uniform(0.01, 0.1)  # 1% to 10% of sender's wealth
            sender.update_wealth(-amount)
            receiver.update_wealth(amount)
            transaction = Transaction(sender.agent_id, receiver.agent_id, amount, i)
            sender.add_transaction(transaction)
            receiver.add_transaction(transaction)
            economy.transactions.append(transaction)

            # update wealth spending
            wealth_spending[sender.agent_id].append(amount)
            if len(wealth_spending[sender.agent_id]) > 100:
                wealth_spending[sender.agent_id].pop(0)
            wealth_spending_counts[sender.agent_id] += 1

        # Apply taxes
        for agent in economy.agents:
            tax = tax_scheme.compute_tax(agent, economy)
            agent.update_wealth(-tax)

        wealth_over_time.append(economy.get_wealth_distribution())

        if i % 10 == 0:  # Update the graph every 10 iterations
            ax1.clear()
            ax1.hist(wealth_over_time[-1], bins=20, color='blue', alpha=0.7)
            ax1.set_title("Wealth Distribution Over Time")

            ax2.clear()
            average_spending = [np.mean(wealth_spending[id]) for id in range(len(economy.agents))]
            ax2.bar(range(len(average_spending)), average_spending, color='green')
            ax2.set_title("Average Spending by Wealth")

            ax3.clear()
            ax3.bar(agent_counts.keys(), agent_counts.values(), color='green')
            ax3.set_title("Agent count at different wealth points")

            plt.draw()
            plt.pause(0.01)

    plt.ioff()
    plt.show()

    # Write the final wealth distribution to a file
    final_wealth_distribution = economy.get_wealth_distribution()
    df = pd.DataFrame(final_wealth_distribution, columns=["Final wealth"])
    df.to_csv("final_wealth_distribution.csv", index=False)

    # Reset economy for the next run
    economy = Economy([Agent(i, initial_wealth) for i in range(num_agents)], [])

    return np.mean(final_wealth_distribution)

if __name__ == "__main__":
    num_agents = 11192
    initial_wealth = 1107.19
    max_iterations = num_agents * 20  # Running 20 times instead of num_agents^2
    agents = [Agent(i, initial_wealth) for i in range(num_agents)]
    economy = Economy(agents, [])
    tax_scheme = TaxScheme(0.1, 0.05, 0.05)
    simulate_economy(economy, tax_scheme, iterations=max_iterations, transactions_per_iteration=num_agents*2)

    bounds = (0, 1)
    start_point = np.array([0.1])
    step_size = 0.01

    num_processors = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processors)

    optimizer = CustomOptimizer(bounds, start_point, step_size)
    optimizer.evaluate = objective_function

    create_dir('winners')
    create_dir('best-winners')

    for _ in range(10000):
        pool.apply_async(optimizer.explore)
    pool.close()
    pool.join()

    # Plotting unsuccessful runs
    plt.barh(['Unsuccessful runs'], [optimizer.unsuccessful_runs], color='white', edgecolor='black')
    plt.text(optimizer.unsuccessful_runs, 0, str(optimizer.unsuccessful_runs), color = 'gray', va='center')
    plt.title('Unsuccessful runs')
    plt.show()

    # Plotting winners
    plt.barh(['Winners'], [len(optimizer.winners)], color='white', edgecolor='black')
    plt.text(len(optimizer.winners), 0, str(len(optimizer.winners)), color = 'gray', va='center')
    plt.title('Winners')
    plt.show()

    # Plotting progress circles
    for point, score in optimizer.memory:
        draw_progress_circles(score, score, score)
