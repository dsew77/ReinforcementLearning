# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:14:30 2021

@author: rscha
"""

#Teaching an agent to play blackjack
#Assume deck of cards infinite (replacement)

#Action space = 2 (simple hit/stay)

#State space = 13*13*(13*13*13)*13
#bracket for the additional cards you can hit

#Rules: Player gets dealt 2 cards, dealer gets delt 1.

import numpy as np
import random
def train_agent():
    #
    max_p_card = 5
    max_cards = (13*np.ones([1,max_p_card+1]).astype(int))[0]
    observation_space = get_state_number(max_cards)
    
    
    q_table = np.zeros([observation_space,2])
    
    #Hyper parameters
    #alpha: is the the learning rate, set generally between 0 and 1. Setting it to 0 means that the Q-values are never updated, thereby nothing is learned. Setting alpha to a high value such as 0.9 means that learning can occur quickly.
    #gamma: is the discount factor, also set between 0 and 1. This models the fact that future rewards are worth less than immediate rewards.
    #Epsilon: exploration vs eploitation 0 and 1
    
    alpha = 0.8
    gamma = 0.2
    epsilon = 0.1
    
    
  
    
    #set up the board
    board = np.zeros([1,max_p_card+1])[0]
    board = board.astype(int)
    
    no_epochs = 200
    fullset = get_card_combo()
    for epoch in range(no_epochs):
        for extract in fullset:
            #Setup dealer and player starting cards
            #Go through each combo (1 epoch is a full set of combos)
            board = np.zeros([1,max_p_card+1])[0]
            board = board.astype(int)
        
        
            board[-3:] = extract
        
        
            times_hit = 0
            done = False
            while not done:
                state = get_state_number(board)
                
            
            
                if random.uniform(0,1) < epsilon:
                    action = int(random.uniform(0,2))  
                elif q_table[state][0] == q_table[state][1]:
                    action = int(random.uniform(0,2))
                else:
                    action = np.argmax(q_table[state])
                            
                    #action 0 = hit
                    #action 1 = stay
           
            
                
            
            
                if action == 1 or times_hit == 3:
                    done = True
                    reward = get_reward(board)
                    old_value = q_table[state,action]
                    new_value = (1-alpha) * old_value + alpha * (reward)
                    q_table[state,action] = new_value
                    
                else:
                    times_hit = times_hit + 1
                    board[-(times_hit + 3)] = get_card()
                        
                    next_state = get_state_number(board)
                        
                    #Check if over 21
                        
                    player, dealer = board_sum(board)
                    if player > 21:
                        reward = -10
                    else:
                        reward = 0
                    old_value = q_table[state,action]
                    next_max = np.max(q_table[next_state])
                    new_value = (1-alpha) * old_value + alpha * (reward + gamma*next_max)
                    q_table[state, action] = new_value
                                
                                
        print('Epoch number' + str(epoch))                                
    return q_table

def get_state_number(cards):
    
    sum = 0
    #Left to right
    
    for i in range(len(cards)):
        #right to left
        #2nd and 3rd cards start at 1
        if i == 2 or i == 3:
            sum = sum + ((cards[-i]-1) * 13**(i-1))
        else:
            sum = sum + (cards[-i] * 13**(i-1))
    return int(sum)


def get_board_from_state_number(statenumber):
    board = np.zeros(6)
    for i in range(6):
        board[i] = np.floor(statenumber/(13**(5-i)))
        statenumber = statenumber % (13**(5-i))
    return board


def get_card():
    #Gets a card between 1-13
    return int(random.uniform(1,13))



def check_random():
    count = np.zeros([1,13])[0].astype(int)
    
    for i in range(10000):
        count[int(random.uniform(1,14))-1] = count[int(random.uniform(1,14))-1] + 1
    return count

def get_reward(board):
    #returns 10 if win
    #returns -10 if lost
    #returns 0 for draw
    player_total, dealer_total = board_sum(board)
    
    #Dealer must draw untill at least 17
    while dealer_total < 17:
        dealer_total = dealer_total + get_card()
        
        
    if player_total > 21:
        return -10
    elif dealer_total > 21:
        return 10
    elif player_total > dealer_total:
        return 10
    elif dealer_total > player_total:
        return -10
    else:
        return 5
        
    
def board_sum(board):
     #Replace 11,12,13 to 10 
    for i in range(len(board)):
        if board[i] > 10:
            board[i] = 10
    
    #returns the player count and the dealer count
    player_cards = board[:-1]
    dealer_cards = [board[-1], get_card()]
    
   
    
    
    player_total = sum(player_cards)
    dealer_total = sum(dealer_cards)
    
    #Check for aces
    
    while player_total <= 11 and 1 in player_cards:
        player_total = player_total + 10
    while dealer_total <= 11 and 1 in dealer_cards:
        dealer_total = dealer_total + 10
    
    return player_total, dealer_total
    

def get_card_combo():
    #return array of 2197 values
    array = np.zeros([2197,3])
    count = 0
    for i in range(13):
        for j in range(13):
            for k in range(13):
                array[count] = [i+1,j+1,k+1]
                count +=1
    
    
    return array.astype(int)

def main(q_table = np.zeros([observation_space,2])):
    
    new_q_table = train_agent(q_table)
    return new_q_table 

def random_agent(**args):
    return int(random(0,2))

def play_blackjack(q_table = 0):
    countw = 0
    countl = 0
    countd = 0
    iteration = 100
    for i in range(iteration):
        #Setup the board
        board = np.zeros([1,6])[0]
        board = board.astype(int)
        board[-3:] = get_card_combo()[int(random.uniform(0,2197))]
        
        
        #Using a random bot
        done = False
        no_hits = 0
        while not done:
            
            #Random bot
            if True:
                action = int(random.uniform(0,2))
                agent = 'random agent'
            
            else:
                #Reinforcement learning bot
                action = np.argmax(q_table[get_state_number(board)])
                agent = 'Learnt AI'
            
            if action == 0 and no_hits <= 2:
                no_hits += 1
                board[-(3+no_hits)] = get_card()
                player, dealer = board_sum(board)
                if player > 21:
                    done = True
                    countl += 1
            
            else:
                reward = get_reward(board)
                if reward >= 10:
                    countw +=1
                elif reward <= -10:
                    countl +=1
                else:
                    countd +=1
                done = True
                

    print('Using ' + agent + ': Times won = ' + str(countw) + ', times lost = ' + str(countl) + ', times draw = ' + str(countd))
























    