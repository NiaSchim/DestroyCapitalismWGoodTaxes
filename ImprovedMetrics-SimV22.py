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
from scipy.stats import norm

threemonths = 2 * 1
onefourthsample = 1  # replace with the actual number

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
    base_rate, wealth_threshold, additional_rate = params
    def calculate_tax(wealth):
        if wealth <= wealth_threshold:
            return additional_rate * np.log1p(max(0, wealth - wealth_threshold))
        else:
            return base_rate * wealth
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
        self.visit_count = defaultdict(int)

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

            # track the visit count of the new point and check if it has been visited 4 times
            self.visit_count[tuple(new_point)] += 1
            if self.visit_count[tuple(new_point)] >= 4*10:
                print(f"Point {new_point} has been visited 4 times. Ending exploration.")
                break

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

def wealth_distribution_to_box_and_whisker(wealth_distribution):
    sorted_wealth = np.sort(wealth_distribution)
    total_population = len(sorted_wealth)
    lower_extreme = np.sum(sorted_wealth == sorted_wealth[0]) / total_population
    lower_quartile = np.sum(sorted_wealth <= np.percentile(sorted_wealth, 25)) / total_population
    median = np.sum((sorted_wealth >= np.percentile(sorted_wealth, 25)) & (sorted_wealth <= np.percentile(sorted_wealth, 50))) / total_population
    upper_quartile = np.sum((sorted_wealth > np.percentile(sorted_wealth, 50)) & (sorted_wealth <= np.percentile(sorted_wealth, 75))) / total_population
    upper_extreme = np.sum(sorted_wealth > np.percentile(sorted_wealth, 75)) / total_population
    return np.array([lower_extreme, lower_quartile, median, upper_quartile, upper_extreme])

def calculate_yardstick(wealthdays):
    box_and_whisker_points = np.array([wealth_distribution_to_box_and_whisker(wealthday) for wealthday in wealthdays])
    max_disparity_point = np.array([0.99, 0, 0, 0, 0.01])
    bell_curve_point = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    return np.max(calculate_distances(box_and_whisker_points))

def calculate_longstraw(wealthdays):
    box_and_whisker_points = np.array([wealth_distribution_to_box_and_whisker(wealthday) for wealthday in wealthdays])
    return np.max(calculate_distances(box_and_whisker_points))

def calculate_bellmarkyardstick(points):
    median_point = np.median(points)
    average_wealth = np.mean(points)
    bell_curve = norm(loc=median_point, scale=average_wealth)
    x = np.linspace(np.min(points), np.max(points), len(points))
    bell_curve_values = bell_curve.pdf(x)
    return np.trapz(bell_curve_values, x)

def calculate_area_under_graph(points):
    return np.trapz(points, dx=1)

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
    wealthdays = [np.nan_to_num(wealthday) for wealthday in wealthdays]

    longstraw = calculate_longstraw(wealthdays)
    yardstick = calculate_yardstick(wealthdays)
    dynamism = longstraw / yardstick

    box_and_whisker_points = np.array([wealth_distribution_to_box_and_whisker(wealthday) for wealthday in wealthdays])
    peoplepoints = np.mean(box_and_whisker_points, axis=0)

    bellmarkyardstick = calculate_bellmarkyardstick(peoplepoints)
    area_under_graph = calculate_area_under_graph(peoplepoints)
    area_of_similarity = bellmarkyardstick - np.abs(area_under_graph - bellmarkyardstick)
    fairity = area_of_similarity / bellmarkyardstick

    GPA = (dynamism + fairity) / 2
    if GPA > 0.5:
        memory.append((point, dynamism, fairity, GPA))

    return GPA

if __name__ == "__main__":
    log = []
    bounds = [(0.1, 1), (1, 12390800), (0.1, 1)]
    start_point = np.random.uniform(0, 1, len(bounds))
    optimizer = CustomOptimizer(bounds, start_point, 0.1)

    optimizer.explore()

    best_point, best_score = optimizer.winners[-1]

    num_agents = 11192
    total_wealth = 12390800
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

