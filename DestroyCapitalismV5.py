#-------------------------------------------------------------------------------
# Name:        Destroying Capitalism with Optimized Progressive Tax, version 4
# Purpose:
#
# Author:      The Schim
#
# Created:     22/07/2023
# Copyright:   (c) The Schim 2023
# Licence:     <BSD3>
#-------------------------------------------------------------------------------
#exact starting conditions loosely inspired by Tuvalu.
import os
import random
import numpy as np
from scipy.optimize import minimize

class Transaction:
    def __init__(self, buyer, seller, amount):
        self.buyer = buyer
        self.seller = seller
        self.amount = amount

class Agent:
    def __init__(self, initial_wealth, votes):
        self.wealth = initial_wealth
        self.votes = votes
        self.history = []
        self.wealth = random.uniform(0, 1)
        self.inheritance = random.uniform(0, 1)
        self.capital_gains = random.uniform(0, 1)
        self.deductions = random.uniform(0, 1)

    def make_decision(self, economy):
        if self.wealth > 0:
            return 'buy'
        else:
            return 'sell'

    def calculate_tax(self, economy):
        total_earned_votes_in_last_30_days = sum(transaction.amount for transaction in self.history if transaction.amount > 0)
        earning_rate = total_earned_votes_in_last_30_days / (30 * 24)
        tax_rate = economy.tax_algorithm(earning_rate, self.wealth, self.inheritance, self.capital_gains, self.deductions)
        positive_history = sum(transaction.amount for transaction in self.history if transaction.amount > 0)
        if tax_rate > 1e100 or positive_history > 1e100:
            return 1e100
        return tax_rate * positive_history

    def receive_redistribution(self, amount):
        self.wealth += amount

class Economy:
    def __init__(self, num_agents, initial_wealth, initial_votes, tax_algorithm):
        self.agents = [Agent(initial_wealth, initial_votes) for _ in range(num_agents)]
        self.tax_algorithm = tax_algorithm

    def calculate_average_votes(self):
        total_votes = sum(agent.votes for agent in self.agents)
        return total_votes / len(self.agents)

    def regenerate_votes(self):
        average_votes = self.calculate_average_votes()
        for agent in self.agents:
            agent.votes += average_votes / (30 * 24)

    def redistribute_tax(self):
        total_tax = sum(agent.calculate_tax(self) for agent in self.agents)
        for agent in self.agents:
            agent.receive_redistribution(total_tax * agent.votes / self.calculate_average_votes())

    def simulate(self):
        previous_total_wealth = self.calculate_total_wealth()
        while True:
            buyer = random.choice(self.agents)
            seller = random.choice(self.agents)
            while buyer == seller:
                seller = random.choice(self.agents)
            amount = random.uniform(0, buyer.wealth)
            if amount > 1e100:
                amount = 1e100
            transaction = Transaction(buyer, seller, amount)
            buyer.wealth -= amount
            seller.wealth += amount
            buyer.history.append(transaction)
            seller.history.append(transaction)
            self.regenerate_votes()
            self.redistribute_tax()
            current_total_wealth = self.calculate_total_wealth()
            if abs(current_total_wealth - previous_total_wealth) < 1e-6:
                break
            previous_total_wealth = current_total_wealth

    def calculate_total_wealth(self):
        return sum(agent.wealth for agent in self.agents)

def tax_algorithm(params):
    marginal_rate, wealth_rate, inheritance_rate, capital_gains_rate, deductions_rate = params
    def calculate_tax(earning_rate, wealth, inheritance, capital_gains, deductions):
        tax = earning_rate * marginal_rate
        tax += wealth * wealth_rate
        tax += inheritance * inheritance_rate
        tax += capital_gains * capital_gains_rate
        tax -= deductions * deductions_rate
        return tax
    return calculate_tax

def objective_function(params):
    economy = Economy(11192, 1107.19, 1107.19, tax_algorithm(params))
    economy.simulate()
    return -economy.calculate_total_wealth()

initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1]
result = minimize(objective_function, initial_guess, method='BFGS')

print('Optimized parameters:', result.x)

if not os.path.exists('winners'):
    os.makedirs('winners')
if not os.path.exists('best-winner'):
    os.makedirs('best-winner')

np.save(os.path.join('winners', 'winner_{}.npy'.format(0)), result.x)

best_winner_index = np.argmin([objective_function(np.load(os.path.join('winners', 'winner_{}.npy'.format(i)))) for i in range(5)])
best_winner = np.load(os.path.join('winners', 'winner_{}.npy'.format(best_winner_index)))

np.save(os.path.join('best-winner', 'best_winner.npy'), best_winner)

print('Best winner:', best_winner)
