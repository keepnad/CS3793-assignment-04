# Daniel Peek qer419
# Michael Canas ohh135
# CS3793 Assignment 04
# 10/19/2018
# neuralnet.py based on Dr. O'Hara's bp.c
# python 3

import random
import neuralnet as nn


# An individual card
class Card:
    def __init__(self, face):
        self.face = face
        if face.isdigit():
            self.value = int(face)
        elif face == "J" or face == "Q" or face == "K":
            self.value = 10
        elif face == "A":
            self.value = 1
        else:
            self.value = -1


# Class for the dealer
class Dealer:
    def __init__(self):
        self.face_up_card = None
        self.hand = []
        self.total = 0
        self.stay = False
        self.bust = False

    # draw a card into the face down hand of the dealer
    def draw(self, deck):
        self.hand.append(deck.pop())
        self.total += self.hand[-1].value
        if self.total >= 17:
            self.stay = True
        if self.total > 21:
            self.bust = True

    # draw a card face up
    def draw_face_up(self, deck):
        self.face_up_card = deck.pop()
        self.total += self.face_up_card.value
        if self.total >= 17:
            self.stay = True

    # reset to initial values, used between games
    def clear(self):
        self.face_up_card = None
        self.hand.clear()
        self.total = 0
        self.stay = False
        self.bust = False


# class for player
class Player:
    def __init__(self):
        self.hand = []
        self.total = 0
        self.stay = False
        self.bust = False

    # draw a card into hand
    def draw(self, deck):
        self.hand.append(deck.pop())
        self.total += self.hand[-1].value
        if self.total > 21:
            self.stay = True
            self.bust = True

    # reset to initial values, used between games
    def clear(self):
        self.hand.clear()
        self.total = 0
        self.stay = False
        self.bust = False


# create deck as a list containing 4 of each card
def make_deck():
    deck = []
    for i in range(4):
        deck.append(Card("2"))
        deck.append(Card("3"))
        deck.append(Card("4"))
        deck.append(Card("5"))
        deck.append(Card("6"))
        deck.append(Card("7"))
        deck.append(Card("8"))
        deck.append(Card("9"))
        deck.append(Card("10"))
        deck.append(Card("J"))
        deck.append(Card("Q"))
        deck.append(Card("K"))
        deck.append(Card("A"))
    return deck


# shuffle the deck with 400 random swaps
def shuffle(deck):
    for i in range(400):
        x = random.randint(0, len(deck) - 1)
        y = random.randint(0, len(deck) - 1)
        deck[x], deck[y] = deck[y], deck[x]


# trainer "cheats" by looking at the top card of the deck to determine ideal play
def desired_output(player, deck):
    if player.total + deck[-1].value > 21:
        # print("current total:", player.total, "next card:", deck[-1].face)
        return 0
    else:
        # print("current total:", player.total, "next card:", deck[-1].face)
        return 1


# run a single game
def play(player, dealer, deck, net_2, net_3):
    global net2_total_right
    global net2_total_wrong
    global net3_total_right
    global net3_total_wrong

    # initial draws that happen every time
    player.draw(deck)
    dealer.draw_face_up(deck)
    player.draw(deck)
    dealer.draw(deck)

    # run the two-card version of the neural net
    # choose to stay or draw and adjust weights based on correctness
    sample = [player.hand[0].value, player.hand[1].value, dealer.face_up_card.value]
    guess = net_2.predict(sample)
    best_choice = desired_output(player, deck)
    net_2.adjust_weights(sample, best_choice)

    # record results
    if guess == best_choice:
        net2_total_right += 1
    else:
        net2_total_wrong += 1

    # set whether or not the net chose to stay
    if guess == 0:
        player.stay = True

    # draw the player's third card
    if player.stay is False:
        player.draw(deck)
    # dealer's third card
    if dealer.stay is False:
        dealer.draw(deck)

    # if the two card net chose to draw, run the three card neural net
    if player.stay is False:
        sample = [player.hand[0].value, player.hand[1].value, player.hand[2].value, dealer.face_up_card.value]
        guess = net_3.predict(sample)
        best_choice = desired_output(player, deck)
        net_3.adjust_weights(sample, best_choice)

        # record results
        if guess == best_choice:
            net3_total_right += 1
        else:
            net3_total_wrong += 1

        if guess == 0:
            player.stay = True

    # draw the player's fourth card
    if player.stay is False:
        player.draw(deck)
    # dealer's fourth card
    if dealer.stay is False:
        dealer.draw(deck)

    # calculate whether or not to draw 5th card
    if player.stay is False:
        # total value and size of a complete deck
        decision_number = 340
        number_of_cards = 52

        # calculate average value of cards in deck
        for card in player.hand:
            decision_number -= card.value
            number_of_cards -= 1
        decision_number -= dealer.face_up_card.value
        number_of_cards -= 1
        decision_number = decision_number / number_of_cards

        # if average value is less than or equal to difference between player's total and 21, draw
        if decision_number <= (21 - player.total):
            player.draw(deck)
            # if total is 21 or under, automatic win by 5 card charlie
            if player.total <= 21:
                return 1
            else:
                return 0
        else:
            player.stay = True

    # dealer's fifth card
    if dealer.stay is False:
        dealer.draw(deck)

    if player.bust is True and dealer.bust is True:
        # print("both bust, dealer wins tie")
        # print("player total:", player.total, "dealer total:", dealer.total, "\n")
        return 0
    elif player.bust is True and dealer.bust is False:
        # print("player busts, dealer wins")
        # print("player total:", player.total, "dealer total:", dealer.total, "\n")
        return 0
    elif player.bust is False and dealer.bust is True:
        # print("dealer busts, player wins")
        # print("player total:", player.total, "dealer total:", dealer.total, "\n")
        return 1
    elif player.bust is False and dealer.bust is False:
        if player.total > dealer.total:
            # print("player wins")
            # print("player total:", player.total, "dealer total:", dealer.total, "\n")
            return 1
        elif player.total < dealer.total:
            # print("dealer wins")
            # print("player total:", player.total, "dealer total:", dealer.total, "\n")
            return 0
        else:
            # print("dealer wins tie")
            # print("player total:", player.total, "dealer total:", dealer.total, "\n")
            return 0


def main():
    global wins
    global losses
    global net2_total_right
    global net2_total_wrong
    global net3_total_right
    global net3_total_wrong

    # set number of games (epochs) and create neural nets
    games = 10000
    two_card_net = nn.NeuralNetwork(3, 12, 2, 0.01)
    three_card_net = nn.NeuralNetwork(4, 16, 2, 0.01)

    # set up players and deck
    random.seed()
    deck = make_deck()
    shuffle(deck)
    player = Player()
    dealer = Dealer()

    # run all games
    for i in range(games):
        # run a single game
        result = play(player, dealer, deck, two_card_net, three_card_net)
        if result == 0:
            losses += 1
        elif result == 1:
            wins += 1

        # reset for next game
        for card in player.hand:
            deck.append(card)
        for card in dealer.hand:
            deck.append(card)
        deck.append(dealer.face_up_card)
        player.clear()
        dealer.clear()
        shuffle(deck)

        # print game number and percent correct choices
        if (i % 1000 == 0) or (i % 5 == 0 and i < 200):
            print("finished game number", i)
            if net2_total_wrong > 0 and net2_total_right > 0:
                two_card_percent = net2_total_right/(net2_total_right + net2_total_wrong)
                print("two card net right choice:   %.02f%%" % (two_card_percent * 100))
                # print("two card net confidence:     %.04f" % two_card_net.confidence)
            if net3_total_wrong > 0 and net3_total_right > 0:
                three_card_percent = net3_total_right/(net3_total_right + net3_total_wrong)
                print("three card net right choice: %.02f%%\n" % (three_card_percent * 100))
                # print("three card net confidence:   %.04f" % three_card_net.confidence)


if __name__ == "__main__":
    wins = 0
    losses = 0
    net2_total_right = 0
    net2_total_wrong = 0
    net3_total_right = 0
    net3_total_wrong = 0
    main()
    # after all games, print total right guesses/wrong guesses and wins/losses
    print("Two card net right", net2_total_right, "wrong", net2_total_wrong)
    print("Three card net right", net3_total_right, "wrong", net3_total_wrong)
    print("wins", wins, "losses", losses)

