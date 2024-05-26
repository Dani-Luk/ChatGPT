import random
from collections import Counter
import matplotlib.pyplot as plt

# Define the card points
card_points = {'A': 4, 'K': 3, 'Q': 2, 'J': 1, '10': 0, '9': 0, '8': 0, '7': 0, '6': 0, '5': 0, '4': 0, '3': 0, '2': 0}

# Define the constant SOUTH_HAND
SOUTH_HAND = {
    'Spades': ['A', 'K', 'Q', '10', '7', '5', '3'],
    'Hearts': ['9', '6', '4'],
    'Diamonds': ['8'],
    'Clubs': ['7', '5']
}

# Function to calculate hand points
def getHandPoints(hand):
    points = 0
    for suit in hand:
        points += sum(card_points[card] for card in hand[suit])
    return points

# Check that getHandPoints(SOUTH_HAND) == 9
assert getHandPoints(SOUTH_HAND) == 9

# Create a full deck and remove SOUTH_HAND cards
full_deck = {
    'Spades': ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'],
    'Hearts': ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'],
    'Diamonds': ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'],
    'Clubs': ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
}

import copy
from collections import OrderedDict

def remove_south_hand(full_deck, south_hand):
    # remaining_deck = full_deck.copy() # This is a shallow copy
    remaining_deck = copy.deepcopy(full_deck)
    for suit in south_hand:
        for card in south_hand[suit]:
            remaining_deck[suit].remove(card)
    return remaining_deck

# Function to deal the rest of the 3 hands
def dealTheRestOf3Hands():
    remaining_deck = remove_south_hand(full_deck, SOUTH_HAND)
    all_remaining_cards = []
    for suit in remaining_deck:
        all_remaining_cards.extend((suit, card) for card in remaining_deck[suit])
    
    while True:
        random.shuffle(all_remaining_cards)
        northHand = {'Spades': [], 'Hearts': [], 'Diamonds': [], 'Clubs': []}
        eastHand = {'Spades': [], 'Hearts': [], 'Diamonds': [], 'Clubs': []}
        westHand = {'Spades': [], 'Hearts': [], 'Diamonds': [], 'Clubs': []}
        
        for i, (suit, card) in enumerate(all_remaining_cards):
            if i % 3 == 0:
                northHand[suit].append(card)
            elif i % 3 == 1:
                eastHand[suit].append(card)
            else:
                westHand[suit].append(card)
        
        if getHandPoints(northHand) <= 11 and getHandPoints(eastHand) <= 11:
            break
    
    return northHand, eastHand, westHand

# Simulate the process 10,000 times and count the occurrences
countOfPossibleOpponentsGame = 0
sumOfEastWestAxis = []

# region GitHub Inline Ctrl+I
# Don't keep every sum_points. Instead use an ordered dictionary for plotting the histogram
# for _ in range(10000):
#     _, eastHand, westHand = dealTheRestOf3Hands()
#     sum_points = getHandPoints(eastHand) + getHandPoints(westHand)
#     sumOfEastWestAxis.append(sum_points)
#     if sum_points >= 24:
#         countOfPossibleOpponentsGame += 1

# # Calculate and print the percentage
# percentage = (countOfPossibleOpponentsGame / 10000) * 100
# print(f"Percentage of possible opponents game: {percentage}%")

# # Plot the histogram
# plt.hist(sumOfEastWestAxis, bins=range(0, 40), edgecolor='black')
# plt.title('Histogram of the Sum of Points of East and West Hands')
# plt.xlabel('Sum of Points')
# plt.ylabel('Frequency')
# plt.show()
# endregion GitHub Inline Ctrl+I

sumOfEastWestAxis = OrderedDict()

for _ in range(10000):
    _, eastHand, westHand = dealTheRestOf3Hands()
    sum_points = getHandPoints(eastHand) + getHandPoints(westHand)
    sumOfEastWestAxis[sum_points] = sumOfEastWestAxis.get(sum_points, 0) + 1
    if sum_points >= 24:
        countOfPossibleOpponentsGame += 1

# Calculate and print the percentage
percentage = (countOfPossibleOpponentsGame / 10000) * 100
print(f"Percentage of possible opponents game: {percentage}%")

# region GitHub Inline Ctrl+I
# show every xLabel on x axis, from min()-1 to max()+1
# Plot the histogram
# plt.bar(sumOfEastWestAxis.keys(), sumOfEastWestAxis.values(), edgecolor='black')
# plt.title('Histogram of the Sum of Points of East and West Hands')
# plt.xlabel('Sum of Points')
# plt.ylabel('Frequency')
# plt.show()
# endregion GitHub Inline Ctrl+I

# Plot the histogram
plt.bar(sumOfEastWestAxis.keys(), sumOfEastWestAxis.values(), edgecolor='black')
plt.title('Histogram of the Sum of Points of East and West Hands')
plt.xlabel('Sum of Points')
plt.ylabel('Frequency')
plt.xticks(range(min(sumOfEastWestAxis.keys()) - 1, max(sumOfEastWestAxis.keys()) + 2))
plt.show()

