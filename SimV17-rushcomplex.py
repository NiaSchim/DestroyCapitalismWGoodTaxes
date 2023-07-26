import os
import numpy as np
import random
from scipy.stats import entropy
from scipy.optimize import minimize
from collections import defaultdict
from scipy.spatial import distance
import pandas as pd
from tqdm import tqdm
import math
from scipy import stats
import time
import inspect
import mpmath
from scipy.spatial.distance import pdist

threemonths = 2 * 1
onefourthsample = 1  # replace with the actual number
system_wide_total_wealth = 12390800

class Agent:
    def __init__(self, agent_id, wealth, personality):
        self.agent_id = agent_id
        self.wealth = wealth
        self.transaction_history = []
        self.wealth_spending = []
        self.positive_history = 0
        self.negative_history = 0
        self.personality = personality
        self.is_richdork = False
        self.is_unfortunate = False
        self.frozen_assets = 0

    def update_wealth(self, amount):
        if self.personality == 'loss_aversion':
            self.wealth += amount * 0.9
        elif self.personality == 'overconfidence':
            self.wealth += amount * 1.1
        elif self.personality == 'altruism':
            self.wealth += amount
        elif self.personality == 'envy':
            if random.random() < 0.5:
                self.wealth += amount * 1.2
            else:
                self.wealth += amount * 0.8
        else:
            self.wealth += amount
        self.wealth = max(0, self.wealth)

    def add_transaction(self, transaction):
        self.transaction_history.append(transaction)

    def update_history(self, purchaser_votes, seller_votes, average_votes, spent_votes):
        if spent_votes != 0:
            self.positive_history = average_votes * (purchaser_votes / spent_votes)
            self.negative_history = average_votes * (seller_votes / spent_votes)
        else:
            self.positive_history = 0
            self.negative_history = 0

class Transaction:
    def __init__(self, sender, receiver, amount, time):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.time = time

class Economy:
    def __init__(self, agents, tax_scheme, start_point):
        self.agents = agents
        self.transactions = []
        self.tax_scheme = tax_scheme
        self.agent_counts_over_time = []
        self.final_wealth_distributions = []
        self.travel_distance = 0
        self.internal_points = 0
        self.treasury = 0
        self.current_point = None
        self.start_point = start_point
        self.wealth_distribution_over_time = []

    def get_wealth_distribution(self):
        return [agent.wealth for agent in self.agents]

    def initial_distribution(self):
        return np.array([agent.wealth for agent in self.agents])

    def perform_taxes(self):
        total_tax = 0
        for agent in self.agents:
            tax = self.tax_scheme(agent.wealth)
            tax = min(tax, agent.wealth)
            agent.update_wealth(-tax)
            total_tax += tax
        self.treasury += total_tax

    def perform_nianomics(self):
        total_wealth = sum([agent.wealth for agent in self.agents]) + self.treasury
        average_wealth = total_wealth / (len(self.agents) + 1)
        for agent in self.agents:
            agent.update_history(agent.wealth, agent.wealth, average_wealth, agent.wealth)
            wealth_adjustment = (average_wealth - agent.wealth) / (threemonths / 3)
            agent.update_wealth(wealth_adjustment)
            if agent.wealth > average_wealth:
                donation_rate = (agent.wealth - average_wealth) / (threemonths / 3)
                agent.update_wealth(-donation_rate)
                self.treasury += donation_rate

    def update_wealthday(self):
        self.wealth_distribution_over_time.append(self.get_wealth_distribution())

    def run_simulation(self):
        num_agents = len(self.agents)
        num_richdorks = random.randint(round(num_agents / 8), round(num_agents * 2 / 3))
        num_unfortunates = random.randint(round(num_agents / 8), round(num_agents * 2 / 3))

        for mode in range(1, 5):
            if mode == 1:
                # Mode 1: No Richdorks, No Unfortunates
                pass
            elif mode == 2:
                # Mode 2: Richdorks, No Unfortunates
                self.introduce_richdorks(num_richdorks)
            elif mode == 3:
                # Mode 3: No Richdorks, Unfortunates
                self.remove_richdorks()
                self.introduce_unfortunates(num_unfortunates)
            elif mode == 4:
                # Mode 4: Richdorks, Unfortunates
                self.introduce_richdorks(num_richdorks)
                self.introduce_unfortunates(num_unfortunates)

            start_time = time.time()
            while time.time() - start_time < onefourthsample:
                self.perform_taxes()
                self.perform_nianomics()
                self.update_wealthday()

    def introduce_richdorks(self, num_richdorks):
        richdorks = random.sample(self.agents, num_richdorks)
        total_wealth = sum(agent.wealth for agent in self.agents)
        starting_wealth_for_richdorks = total_wealth * random.uniform(0.5, 0.9)
        for agent in richdorks:
            extra_wealth = starting_wealth_for_richdorks / num_richdorks
            agent.update_wealth(extra_wealth)
            agent.is_richdork = True

    def remove_richdorks(self):
        for agent in self.agents:
            if agent.is_richdork:
                agent.is_richdork = False

    def introduce_unfortunates(self, num_unfortunates):
        unfortunates = random.sample([agent for agent in self.agents if not agent.is_richdork], num_unfortunates)
        mode_wealth = stats.mode([agent.wealth for agent in self.agents])[0][0]
        for agent in unfortunates:
            debt = mode_wealth * random.uniform(0.2, 1.0)
            agent.update_wealth(-debt)
            agent.frozen_assets = debt
            agent.is_unfortunate = True

def tax_algorithm(params):
    lower_power, translation_ammount, higher_power = params
    def calculate_tax(wealth):
        if (wealth - translation_ammount)**((higher_power/system_wide_total_wealth)+1) > wealth**(lower_power/system_wide_total_wealth):
            return (wealth - translation_ammount)**((higher_power/system_wide_total_wealth)+1)
        else:
            return wealth**(lower_power/system_wide_total_wealth)
    return calculate_tax

class CustomOptimizer:
    def __init__(self, bounds, start_point, step_size):
        self.bounds = bounds
        self.start_point = np.array(start_point)
        self.step_size = step_size
        self.memory = []
        self.unsuccessful_runs = 0
        self.consecutive_unsuccessful_runs = 0
        self.winners = []
        self.current_point = np.array(start_point)
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
            file.write(inspect.getsource(tax_algorithm).replace('params', str(point)))

    def evaluate(self, point):
        return objective_function(point, self.memory)

    def explore(self):
        while self.consecutive_unsuccessful_runs < self.max_consecutive_failures:
            force_vector = np.zeros_like(self.current_point)
            if self.no_improvement_streak > 0:
                direction_to_start = self.start_point - self.current_point
                force_vector += direction_to_start * (self.no_improvement_streak / 10.0)

            for winner in self.winners:
                direction_to_winner = np.array(winner[0]) - self.current_point
                norm = np.linalg.norm(direction_to_winner)
                if norm != 0 and not np.any(np.isnan(direction_to_winner)):
                    force_vector -= direction_to_winner / norm

            step_vector = self.random_step() + force_vector
            lower_bounds, upper_bounds = np.array(self.bounds).T
            new_point = self.current_point + step_vector
            new_point = np.clip(new_point, lower_bounds, upper_bounds)
            new_score = self.evaluate(new_point)

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

            if self.consecutive_unsuccessful_runs >= 10:
                if self.winners:
                    random_winner = random.choice(self.winners)
                    self.current_point, self.current_score = random_winner
                    self.consecutive_unsuccessful_runs = 0

            self.memory.append((self.current_point, self.current_score))

            if self.iterations % 10 == 0:
                print("\nIteration:", self.iterations)
                print("Current Point:", self.current_point)
                print("Current Score:", self.current_score)

            self.iterations += 1

def calculate_distances(points):
    return pdist(points, 'euclidean')

def calculate_shortstraw(points):
    distances = calculate_distances(points)
    return np.min(distances)

def calculate_yardstick(points):
    num_points = len(points)
    sphere_points = np.random.normal(size=(num_points, 6))
    sphere_points /= np.linalg.norm(sphere_points, axis=1)[:, np.newaxis]
    sphere_distances = calculate_distances(sphere_points)
    return np.mean(sphere_distances)


def objective_function(point, memory):
    num_agents = 11192
    total_wealth = 12390800
    initial_wealth_per_agent = total_wealth / num_agents
    personalities = ['loss_aversion', 'overconfidence', 'altruism', 'envy', 'default']
    agents = [Agent(i, initial_wealth_per_agent, random.choice(personalities)) for i in range(num_agents)]
    economy = Economy(agents, tax_algorithm(point), start_point)

    start_time = time.time()
    while time.time() - start_time < threemonths:
        economy.perform_taxes()
        economy.perform_nianomics()
        economy.update_wealthday()

    wealthdays = economy.wealth_distribution_over_time
    wealthdays = [np.array(wealthday) for wealthday in wealthdays]
    wealthdays = [wealthday / np.linalg.norm(wealthday) if np.linalg.norm(wealthday) != 0 else wealthday for wealthday in wealthdays]
    wealthdays = [np.nan_to_num(wealthday) for wealthday in wealthdays]  # replace 'nan' with 0 and 'inf' with large finite numbers
    shortstraw = calculate_shortstraw(wealthdays)
    yardstick = calculate_yardstick(wealthdays)
    dynamism = shortstraw / yardstick

    peoplepoints = [np.histogram(wealthday, bins=100)[0] for wealthday in wealthdays]
    peoplepoints = [peoplepoint / np.linalg.norm(peoplepoint) for peoplepoint in peoplepoints]
    peoplepoints = np.mean(peoplepoints, axis=0)
    midpoint = (np.max(peoplepoints) + np.min(peoplepoints)) / 2
    deviations = np.abs(peoplepoints - midpoint)
    fairity = 1 - (np.sum(deviations) / (len(peoplepoints) * np.max(deviations)))

    GPA = (dynamism + fairity) / 2
    if GPA > 0.005:
        memory.append((point, dynamism, fairity, GPA))

    return GPA

if __name__ == "__main__":
    log = []
    bounds = [(0, 1), (0, 1000), (0, 1)]
    start_point = np.random.uniform(0, 1, len(bounds))
    optimizer = CustomOptimizer(bounds, start_point, 0.1)

    optimizer.explore()

    best_point, best_score = optimizer.winners[-1]

    num_agents = 11192
    total_wealth = system_wide_total_wealth
    initial_wealth_per_agent = total_wealth / num_agents
    personalities = ['loss_aversion', 'overconfidence', 'altruism', 'envy', 'default']
    agents = [Agent(i, initial_wealth_per_agent, random.choice(personalities)) for i in range(num_agents)]
    economy = Economy(agents, tax_algorithm(best_point), start_point=best_point)

    start_time = time.time()
    tax_interval = threemonths / 3
    while time.time() - start_time < threemonths:
        economy.perform_taxes()
        economy.perform_nianomics()
        economy.update_wealthday()

    wealthdays = economy.wealth_distribution_over_time
    wealthdays = [np.array(wealthday) for wealthday in wealthdays]
    wealthdays = [wealthday / np.linalg.norm(wealthday) for wealthday in wealthdays]
    shortstraw = calculate_shortstraw(wealthdays)
    yardstick = calculate_yardstick(wealthdays)
    dynamism = shortstraw / yardstick

    peoplepoints = [np.histogram(wealthday, bins=100)[0] for wealthday in wealthdays]
    peoplepoints = [peoplepoint / np.linalg.norm(peoplepoint) for peoplepoint in peoplepoints]
    peoplepoints = np.mean(peoplepoints, axis=0)
    midpoint = (np.max(peoplepoints) + np.min(peoplepoints)) / 2
    deviations = np.abs(peoplepoints - midpoint)
    fairity = 1 - (np.sum(deviations) / (len(peoplepoints) * np.max(deviations)))

    GPA = (dynamism + fairity) / 2
    print("Wealth Dynamism Grade: {:.2f} out of 100%".format(dynamism * 100))
    print("Wealth Pattern Avoidance Grade: {:.2f} out of 100%".format(fairity * 100))
    print("GPA: {:.2f}".format(GPA))

