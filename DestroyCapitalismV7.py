import os
import numpy as np
import random
from scipy.stats import entropy

class CustomOptimizer:
    def __init__(self, bounds, start_point, step_size):
        self.bounds = bounds
        self.start_point = start_point
        self.step_size = step_size
        self.memory = []
        self.unsuccessful_runs = 0
        self.consecutive_unsuccessful_runs = 0
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
            self.consecutive_unsuccessful_runs = 0
            if len(self.winners) == 0 or new_score > self.winners[0][1]:
                self.winners = [(new_point, new_score)]
                self.save_winner(new_point, new_score, 'best-winners')
            elif new_score == self.winners[0][1]:
                self.winners.append((new_point, new_score))
            self.save_winner(new_point, new_score, 'winners')
        else:
            self.no_improvement_streak += 1
            self.unsuccessful_runs += 1
            self.consecutive_unsuccessful_runs += 1
            if np.array_equal(self.current_point, self.start_point):
                self.current_point = np.random.uniform(self.bounds[0], self.bounds[1], len(self.current_point))
                if self.consecutive_unsuccessful_runs >= 10:
                    self.compare_winners()
        self.memory.append((new_point, new_score))

        print(f"Total failed runs: {self.unsuccessful_runs}")
        print(f"Consecutive failed runs: {self.consecutive_unsuccessful_runs}")
        print(f"Total number of winners: {len(self.winners)}")

    def compare_winners(self):
        best_score = max([score for _, score in self.winners])
        best_winners = [winner for winner, score in self.winners if score == best_score]
        for i, winner in enumerate(best_winners):
            self.save_winner(winner, best_score, 'best-winners')

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
    def __init__(self, num_agents, tax_scheme):
        self.agents = [Agent(i, 100) for i in range(num_agents)]
        self.transactions = []
        self.tax_scheme = tax_scheme

    def get_wealth_distribution(self):
        return [agent.wealth for agent in self.agents]

def tax_algorithm(params):
    base_rate, wealth_threshold, additional_rate = params
    def calculate_tax(wealth):
        if wealth > wealth_threshold:
            return base_rate * wealth + additional_rate * (wealth - wealth_threshold)
        else:
            return base_rate * wealth
    return calculate_tax

def simulate_economy(economy, iterations=1000, transactions_per_iteration=100, num_runs=20):
    wealth_over_time = []
    for _ in range(num_runs):
        for i in range(iterations):
            for _ in range(transactions_per_iteration):
                sender = random.choice(economy.agents)
                receiver = random.choice(economy.agents)
                while receiver == sender:
                    receiver = random.choice(economy.agents)
                amount = random.uniform(0, sender.wealth)
                sender.update_wealth(-amount)
                receiver.update_wealth(amount)
                transaction = Transaction(sender, receiver, amount, i)
                sender.add_transaction(transaction)
                receiver.add_transaction(transaction)
                economy.transactions.append(transaction)
            wealth_over_time.append(economy.get_wealth_distribution())
    return wealth_over_time

def objective_function(params, log):
    total_entropy = 0
    for _ in range(5):
        economy = Economy(100, tax_algorithm(params))
        wealth_distribution = simulate_economy(economy)
        wealth_distribution = np.histogram(wealth_distribution[-1], bins=100)[0]
        wealth_distribution = wealth_distribution / np.sum(wealth_distribution)
        total_entropy += entropy(wealth_distribution)
    average_entropy = total_entropy / 5
    log.append((params, average_entropy))
    return average_entropy

if __name__ == "__main__":
    log = []
    bounds = [(0, 1), (0, 1), (0, 1)]
    start_point = [0.5, 0.5, 0.5]
    step_size = 0.1
    optimizer = CustomOptimizer(bounds, start_point, step_size)
    optimizer.explore()
    print(log)
