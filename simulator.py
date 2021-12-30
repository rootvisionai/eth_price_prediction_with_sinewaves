# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:56:05 2021

@author: tekin.evrim.ozmermer
"""

class Buyer(object):
    def __init__(self, prediction_data, real_data, usd = 10000, eth = 0, r = 40, amount = 1):
        self.prediction_data = prediction_data
        self.real_data = real_data
        self.usd  = usd
        self.eth  = eth
        self.initial_capital = self.real_data[0]*eth + usd
        self.capital = {-1:self.initial_capital}
        self.r = r
        self.amount = amount
        
    def decide(self, turn):
        if (self.prediction_data[turn:turn+self.r]).max() > self.prediction_data[turn]:
            self.decision = "buy"
        else:
            self.decision = "sell"
        
    def buy(self, turn, amount=1, multiplier = None):
        
        if self.usd>=amount:
            self.usd -= amount
            self.eth += amount/(self.real_data[turn]+0.00001)

    def sell(self, turn, amount=1, multiplier = None):
        
        if self.eth>=amount:
            self.eth -= amount
            self.usd += self.real_data[turn]*amount
        
    def simulate(self):
        for turn in range(self.prediction_data.shape[0]-(self.r)):
            if turn == self.prediction_data.shape[0]-(self.r+1):
                print("___\n")
                print(self.real_data[turn], self.eth)
                print("\n___")
                self.sell(turn, amount=self.amount)
            else:
                self.decide(turn)
                if self.decision == "buy":
                    self.buy(turn, amount=self.amount)
                elif self.decision == "sell":
                    self.sell(turn, amount=self.amount)
            cap = self.real_data[0]*self.eth + self.usd
            print("TURN: ", turn, "CAPITAL: ", cap.item(),
                  "DECISION: ", self.decision,
                  "USD:", self.usd, "ETH:", self.eth)
            
            self.capital[turn] = cap
        print("\nSIMULATION FINISHED --->")
        print("Initial: ", self.initial_capital)
        print("Final: "  , self.capital[turn])
        return self.capital