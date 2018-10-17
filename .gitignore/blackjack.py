import random
import neuralnet as nn

net2_total_right = 0
net2_total_wrong = 0
net3_total_right = 0
net3_total_wrong = 0

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


class Dealer:
    def __init__(self):
        self.face_up_card = None
        self.hand = []
        self.total = 0
        self.stay = False

    def draw(self, deck):
        self.hand.append(deck.pop())
        self.total += self.hand[-1].value
        if self.total >= 17:
            self.stay = True

    def draw_face_up(self, deck):
        self.face_up_card = deck.pop()
        self.total += self.face_up_card.value
        if self.total >= 17:
            self.stay = True


class Player:
    def __init__(self):
        self.hand = []
        self.total = 0
        self.stay = False

    def draw(self, deck):
        self.hand.append(deck.pop())
        self.total += self.hand[-1].value


def make_deck():
    # create deck as a list containing 4 of each card
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


def shuffle(deck):
    for i in range(10000):
        x = random.randint(0, len(deck) - 1)
        y = random.randint(0, len(deck) - 1)
        deck[x], deck[y] = deck[y], deck[x]


def desired_output(player, deck):
    if player.total + deck[-1].value > 21:
        return 0
    else:
        return 1


def play(player, dealer, deck, net_2, net_3):
    global net2_total_right
    global net2_total_wrong
    global net3_total_right
    global net3_total_wrong

    player.draw(deck)
    dealer.draw_face_up(deck)
    player.draw(deck)
    dealer.draw(deck)

    sample = [player.hand[0].value, player.hand[1].value, dealer.face_up_card.value]
    guess = net_2.predict(sample)
    best_choice = desired_output(player, deck)
    net_2.adjust_weights(sample, best_choice)

    if guess == best_choice:
        net2_total_right += 1
    else:
        net2_total_wrong += 1

    if guess == 0:
        player.stay = True
    else:
        player.draw(deck)
        if dealer.stay is False:
            dealer.draw(deck)

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
        else:
            player.draw(deck)
            if dealer.stay is False:
                dealer.draw(deck)


def main():
    # set number of games and create neural nets
    games = 100
    two_card_net = nn.NeuralNetwork(3, 4, 2, 0.1)
    three_card_net = nn.NeuralNetwork(4, 4, 2, 0.1)

    # set up players and deck
    random.seed()
    deck = make_deck()
    shuffle(deck)
    player = Player()
    dealer = Dealer()

    # run all games
    for i in range(games):
        # run a single game
        play(player, dealer, deck, two_card_net, three_card_net)

        # reset for next game
        for card in player.hand:
            deck.append(card)
        for card in dealer.hand:
            deck.append(card)
        deck.append(dealer.face_up_card)
        player.hand.clear()
        dealer.hand.clear()
        shuffle(deck)

        # print game number
        print(i)


if __name__ == "__main__":
    main()
    # after all games, print total right and wrong guesses
    print("net 2 right", net2_total_right, "wrong", net2_total_wrong)
    print("net 3 right", net3_total_right, "wrong", net3_total_wrong)

