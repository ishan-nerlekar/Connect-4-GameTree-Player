# /usr/bin/env python3
from FourConnect import *  # See the FourConnect.py file
import copy
import csv
from tqdm import tqdm


class GameTreePlayer:
    def __init__(self, max_depth, player_color=2):
        self.max_depth = max_depth
        self.player_color = player_color

    # Function to find the best possible action for GameTreePlayer based on the Minimax Algorithm
    def FindBestAction(self, four_connect):
        current_state = four_connect.GetCurrentState()
        legal_actions = self.GetLegalActions(current_state)

        if not legal_actions:
            return None  # No legal actions available

        best_action = self.Minimax(
            current_state, self.max_depth, float("-inf"), float("inf"), True)[1]

        return best_action

    # Evaluation heuristic 1: The computer will choose the highest utlity value it finds. It will choose the corresponding column to play its next move. It will first check if it can get consecutive fours, then threes and lastly twos. Here we calculate the number of possible fours, threes and twos that the GameTreePlayer can make and subtract them from the opponent.
    def Evaluate(self, state):
        my_color = 2  # Assuming GameTreePlayer is always Player 2
        opponent_color = 1 if my_color == 2 else 2

        fours = self.checkForStreak(state, my_color, 4)
        threes = self.checkForStreak(state, my_color, 3)
        twos = self.checkForStreak(state, my_color, 2)

        opp_fours = self.checkForStreak(state, opponent_color, 4)
        opp_threes = self.checkForStreak(state, opponent_color, 3)
        opp_twos = self.checkForStreak(state, opponent_color, 2)

        return (fours * 10 + threes * 5 + twos * 2) - (opp_fours * 10 + opp_threes * 5 + opp_twos * 2)

    # Evaluation heauristic 2: Players are scored based on favourable positions captured on the board.
    # def Evaluate(self, state):
    #     return self.evaluate_custom(state)

    def evaluate_custom(self, state):
        my_color = 2  # Assuming Game Tree player is always Player 2
        opponent_color = 1 if my_color == 2 else 2

        my_score = 0
        opponent_score = 0

        for i in range(6):
            for j in range(7):
                if state[i][j] == my_color:
                    my_score += self.evaluate_position(i, j, state, my_color)
                elif state[i][j] == opponent_color:
                    opponent_score += self.evaluate_position(
                        i, j, state, opponent_color)

        return my_score - opponent_score

    def evaluate_position(self, row, col, state, color):
        score = 0

        # Add points for controlling the center column
        if col == 3:
            score += 2

        # Add points for having consecutive pieces in a row
        row_streak = self.count_streak(row, col, state, color, (0, 1), 0)
        if row_streak >= 3:
            score += row_streak * 2

        # Add points for having consecutive pieces in a column
        col_streak = self.count_streak(row, col, state, color, (1, 0), 0)
        if col_streak >= 3:
            score += col_streak * 2

        return score

    def count_streak(self, row, col, state, color, direction, count):
        if not (0 <= row < 6) or not (0 <= col < 7) or state[row][col] != color:
            return count
        return self.count_streak(row + direction[0], col + direction[1], state, color, direction, count + 1)

    # This is a simple evaluation function that only considers the number of pieces on the board without taking into account their positions or any strategic considerations
    # def Evaluate(self, state):
    #     my_color = 2  # Assuming Game Tree player is always Player 2
    #     opponent_color = 1 if my_color == 2 else 2

    #     my_count = sum(row.count(my_color) for row in state)
    #     opponent_count = sum(row.count(opponent_color) for row in state)

    #     return my_count - opponent_count

    # Returns a list of all columns which can be played into
    def GetLegalActions(self, state):
        return [col for col in range(7) if state[0][col] == 0]

    # Checks for any streak of any color of specified length
    def checkForStreak(self, state, color, streak):
        count = 0
        for i in range(6):
            for j in range(7):
                if state[i][j] == color:
                    count += self.verticalStreak(i, j, state, streak)
                    count += self.horizontalStreak(i, j, state, streak)
                    count += self.diagonalCheck(i, j, state, streak)

        return count

    # Checks for any vertical streak of any color of specified length
    def verticalStreak(self, row, column, state, streak):
        consecutiveCount = 0
        for i in range(row, 6):
            if state[i][column] == state[row][column]:
                consecutiveCount += 1
            else:
                break
        return consecutiveCount >= streak

    # Checks for any horizontal streak of any color of specified length
    def horizontalStreak(self, row, column, state, streak):
        count = 0
        for j in range(column, 7):
            if state[row][j] == state[row][column]:
                count += 1
            else:
                break
        return count >= streak

    # Checks for any diagonal streak of any color of specified length
    def diagonalCheck(self, row, column, state, streak):
        total = 0
        count = 0
        j = column
        for i in range(row, 6):
            if j > 6:
                break
            elif state[i][j] == state[row][column]:
                count += 1
            else:
                break
            j += 1
        if count >= streak:
            total += 1
        count = 0
        j = column
        for i in range(row, -1, -1):
            if j > 6:
                break
            elif state[i][j] == state[row][column]:
                count += 1
            else:
                break
            j += 1
        if count >= streak:
            total += 1
        return total

    # Minimax function to calculate the best possible move, implementing alpha-beta pruning
    def Minimax(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.IsTerminal(state):
            return self.Evaluate(state), None

        legal_actions = self.GetLegalActions(state)

        if maximizing_player:
            value = float("-inf")
            best_action = None
            for action in legal_actions:
                new_state = self.GetResult(state, action, 2)
                if new_state is not None:
                    evaluation, _ = self.Minimax(
                        new_state, depth - 1, alpha, beta, False
                    )
                    if evaluation is not None and evaluation > value:
                        value = evaluation
                        best_action = action
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
            return value, best_action
        else:
            value = float("inf")
            best_action = None
            for action in legal_actions:
                new_state = self.GetResult(state, action, 1)
                if new_state is not None:
                    evaluation, _ = self.Minimax(
                        new_state, depth - 1, alpha, beta, True
                    )
                    if evaluation is not None and evaluation < value:
                        value = evaluation
                        best_action = action
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
            return value, best_action

    # Minimax with a move ordering function to calculate the best possible move, implementing alpha-beta pruning
    def Minimax_with_move_ordering(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.IsTerminal(state):
            return self.Evaluate(state), None

        legal_actions = self.GetLegalActions(state)

        # Implement move ordering based on evaluation
        sorted_actions = sorted(
            legal_actions,
            key=lambda action: self.Evaluate(self.GetResult(state, action, 2)),
            reverse=maximizing_player  # For maximizing player, sort in descending order
        )

        if maximizing_player:
            value = float("-inf")
            best_action = None
            for action in sorted_actions:
                new_state = self.GetResult(state, action, 2)
                if new_state is not None:
                    evaluation, _ = self.Minimax(
                        new_state, depth - 1, alpha, beta, False
                    )
                    if evaluation is not None and evaluation > value:
                        value = evaluation
                        best_action = action
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
            return value, best_action
        else:
            value = float("inf")
            best_action = None
            for action in sorted_actions:
                new_state = self.GetResult(state, action, 1)
                if new_state is not None:
                    evaluation, _ = self.Minimax(
                        new_state, depth - 1, alpha, beta, True
                    )
                    if evaluation is not None and evaluation < value:
                        value = evaluation
                        best_action = action
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
            return value, best_action

    # Function to check if the game is in a terminal state
    def IsTerminal(self, state):
        return (
            self.checkForStreak(state, 1, 4)
            or self.checkForStreak(state, 2, 4)
            or all(state[0])
        )

    # Function to take action based on the evaluated value of the board
    def GetResult(self, state, action, player):
        new_state = copy.deepcopy(state)
        self.EditBoard(new_state, action, player)
        return new_state

    # Function to edit the board and make the changes based on the action taken
    def EditBoard(self, state, action, player):
        if state is None or action is None or not isinstance(action, int):
            return None
        row = self.RowCheck(state, action)
        if row is not None and isinstance(row, int):
            state[row][action] = player
        else:
            print("Error: Unable to determine a valid row.")

    # Function to find topmost available row in a particular column
    def RowCheck(self, state, action):
        empty_row = -1
        for row in reversed(range(6)):
            if state[row][action] == 0:
                empty_row = row
                break
        return empty_row


def LoadTestcaseStateFromCSVfile():
    testcase_state = list()

    with open("testcase_hard1.csv", "r") as read_obj:
        csv_reader = csv.reader(read_obj)
        for csv_row in csv_reader:
            row = [int(r) for r in csv_row]
            testcase_state.append(row)
        return testcase_state


def PlayGame():
    myopic_player = FourConnect()
    game_tree_player = GameTreePlayer(max_depth=5, player_color=2)
    initial_state = LoadTestcaseStateFromCSVfile()
    myopic_player.SetCurrentState(initial_state)
    # myopic_player.PrintGameState(myopic_player.GetCurrentState())
    total_moves = 0
    game_tree_player_moves = 0

    while total_moves < 42:  # At most 42 moves are possible
        if total_moves % 2 == 0:  # Myopic player always moves first
            myopic_player.MyopicPlayerAction()
            if myopic_player.winner:
                print("Myopic Player wins!")
                break
        else:
            currentState = myopic_player.GetCurrentState()
            gameTreeAction = game_tree_player.FindBestAction(
                myopic_player)
            game_tree_player_moves += 1
            if gameTreeAction is not None:
                myopic_player.GameTreePlayerAction(gameTreeAction)
                currentState = myopic_player.GetCurrentState()
                if game_tree_player.checkForStreak(currentState, 2, 4) > 0:
                    print("GameTree Player wins!")
                    break
            else:
                print("Game Tree Player chose an invalid action. Skipping.")
        myopic_player.PrintGameState()
        total_moves += 1

        if myopic_player.winner is not None:
            break

    # myopic_player.PrintGameState(myopic_player.GetCurrentState())
    print("Moves: {}".format(total_moves))
    print("GameTreePlayer Moves: {}".format(game_tree_player_moves))


# def RunTestCase():
#     fourConnect = FourConnect()
#     gameTree = GameTreePlayer(max_depth=5, player_color=2)
#     testcaseState = LoadTestcaseStateFromCSVfile()
#     fourConnect.SetCurrentState(testcaseState)
#     fourConnect.PrintGameState(testcaseState)
#     gameTreeAction = gameTree.FindBestAction(fourConnect)
#     print("2020B3A71515G", gameTreeAction)

def RunTestCase():
    """
    This procedure reads the state in testcase.csv file and start the game.
    Player 2 moves first. Player 2 must win in 5 moves to pass the testcase; Otherwise, the program fails to pass the testcase.
    """

    fourConnect = FourConnect()
    gameTree = GameTreePlayer(max_depth=5, player_color=2)
    testcaseState = LoadTestcaseStateFromCSVfile()
    fourConnect.SetCurrentState(testcaseState)
    fourConnect.PrintGameState()

    move = 0
    while move < 5:  # Player 2 must win in 5 moves
        if move % 2 == 1:
            fourConnect.MyopicPlayerAction()
        else:
            currentState = fourConnect.GetCurrentState()
            gameTreeAction = gameTree.FindBestAction(fourConnect)
            fourConnect.GameTreePlayerAction(gameTreeAction)
        fourConnect.PrintGameState()
        move += 1
        if fourConnect.winner != None:
            break

    print("Roll no : 2020B3A71515G")  # Put your roll number here

    if fourConnect.winner == 2:
        print("Player 2 has won. Testcase passed.")
    else:
        print("Player 2 could not win in 5 moves. Testcase failed.")
    print("Moves : {0}".format(move))


def main():
    # for i in range(50):
    #     PlayGame()
    # PlayGame()

    RunTestCase()  # Uncomment this line when running the test case


if __name__ == "__main__":
    main()
