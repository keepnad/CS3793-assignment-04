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


# shuffle the deck with 10000 random swaps
def shuffle(deck):
    for i in range(10000):
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

    # for card in player.hand:
    #     print(card.face)

    # run the two-card version of the neural net
    sample = [player.hand[0].value, player.hand[1].value, dealer.face_up_card.value]
    guess = net_2.predict(sample)
    best_choice = desired_output(player, deck)
    net_2.adjust_weights(sample, best_choice)

    # record results
    if guess == best_choice:
        net2_total_right += 1
    else:
        net2_total_wrong += 1

    if guess == 0:
        player.stay = True

    if dealer.stay is False:
        dealer.draw(deck)

    if player.stay is False:
        player.draw(deck)
        # for card in player.hand:
        #     print(card.face)
        if player.total > 21:
            player.bust = True
            player.stay = True

    if player.stay is False:
        sample = [player.hand[0].value, player.hand[1].value, player.hand[2].value, dealer.face_up_card.value]
        guess = net_3.predict(sample)
        best_choice = desired_output(player, deck)
        net_3.adjust_weights(sample, best_choice)

        if guess == best_choice:
            net3_total_right += 1
        else:
            net3_total_wrong += 1

        if guess == 0:
            player.stay = True

    if player.stay is False:
        player.draw(deck)
        # for card in player.hand:
        #     print(card.face)
        if player.total > 21:
            player.bust = True
            player.stay = True

    while dealer.stay is False:
        dealer.draw(deck)

    if player.bust is True and dealer.bust is True:
        print("both bust, dealer wins tie")
        print("player total:", player.total, "dealer total:", dealer.total, "\n")
        return 0
    elif player.bust is True and dealer.bust is not True:
        print("player busts, dealer wins")
        print("player total:", player.total, "dealer total:", dealer.total, "\n")
        return 0
    elif player.bust is False and dealer.bust is True:
        print("dealer busts, player wins")
        print("player total:", player.total, "dealer total:", dealer.total, "\n")
        return 1
    elif player.bust is False and dealer.bust is False:
        if player.total > dealer.total:
            print("player wins")
            print("player total:", player.total, "dealer total:", dealer.total, "\n")
            return 1
        elif player.total < dealer.total:
            print("dealer wins")
            print("player total:", player.total, "dealer total:", dealer.total, "\n")
            return 0
        else:
            print("dealer wins tie")
            print("player total:", player.total, "dealer total:", dealer.total, "\n")
            return 0


def main():
    global wins
    global losses
    # set number of games and create neural nets
    games = 10
    two_card_net = nn.NeuralNetwork(3, 8, 2, 0.1)
    three_card_net = nn.NeuralNetwork(4, 8, 2, 0.1)

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
        player.__init__()
        dealer.__init__()
        shuffle(deck)

        # print game number
        # print("finished game number", i)


if __name__ == "__main__":
    wins = 0
    losses = 0
    net2_total_right = 0
    net2_total_wrong = 0
    net3_total_right = 0
    net3_total_wrong = 0
    main()
    # after all games, print total right and wrong guesses
    print("net 2 right", net2_total_right, "wrong", net2_total_wrong)
    print("net 3 right", net3_total_right, "wrong", net3_total_wrong)
    print("wins", wins, "losses", losses)

