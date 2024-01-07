import random
import copy

class FourConnect:
    
    def __init__(self):
        self._47_game = [
                    [0,0,0,0,0,0,0], #row 0 having columns 0 to 6 from left to right
                    [0,0,0,0,0,0,0], #row 1 having columns 0 to 6 from left to right
                    [0,0,0,0,0,0,0], #row 2 having columns 0 to 6 from left to right
                    [0,0,0,0,0,0,0], #row 3 having columns 0 to 6 from left to right
                    [0,0,0,0,0,0,0], #row 4 having columns 0 to 6 from left to right
                    [0,0,0,0,0,0,0]  #row 5 having columns 0 to 6 from left to right
                    ]
        self.winner = None
    
    def _47_CoinRowAfterAction(self,action):
        g = self._47_game
        cRow = -1
        c=action
        for r in range(5,-1,-1):
            if g[r][c]==0:
                cRow=r
                break
        return cRow

    def _47_CheckHorizontal(self,row,col,player):
        g = self._47_game
        cMin=cMax=col
        while True:
            cMin = cMin-1
            if cMin<0 or g[row][cMin]!=player:
                cMin = cMin+1
                break
        while True:
            cMax = cMax+1
            if cMax>6 or g[row][cMax]!=player:
                cMax = cMax-1
                break
        if cMax-cMin>=3:
            return True
        else:
            return False

    def _47_CheckVertical(self,row,col,player):
        g = self._47_game
        rMin=rMax=row
        while True:
            rMin = rMin-1
            if rMin<0 or g[rMin][col]!=player:
                rMin = rMin+1
                break
        while True:
            rMax = rMax+1
            if rMax>5 or g[rMax][col]!=player:
                rMax = rMax-1
                break
        if rMax-rMin>=3:
            return True
        else:
            return False


    def _47_CheckDiag(self,row,col,player,diag=1):
        g = self._47_game
        cMin=cMax=col
        rMin=rMax=row
        while True:
            cMin = cMin -1
            rMax = rMax + diag
            if cMin<0 or rMax<0 or rMax>5 or g[rMax][cMin]!=player:
                cMin = cMin +1
                rMax = rMax - diag
                break
        while True:
            cMax = cMax+1
            rMin = rMin-diag
            if cMax>6 or rMin<0 or rMin>5 or g[rMin][cMax]!=player:
                cMax = cMax-1
                rMin = rMin+diag
                break
        if cMax-cMin>=3:
            return True
        else:
            return False

        
    def _47_CanAPlayerWin(self,row,col,player):
        win = self._47_CheckHorizontal(row,col,player)
        if win==False:
            win = self._47_CheckDiag(row,col,player,diag=+1) #Principal diagonal = +1
        if win==False:
            win = self._47_CheckDiag(row,col,player,diag=-1) #Secondary diagonal = -1
        if win==False:
            win = self._47_CheckVertical(row,col,player)
        return win

    def _47_CanMyopicPlayerWin(self,row,col):
        return self._47_CanAPlayerWin(row,col,player=1)

    def _47_CanGameTreePlayerWin(self,row,col):
        return self._47_CanAPlayerWin(row,col,player=2)


    def _47_FindMyopicMoves(self):
        validAction=list()
        losingAction=list()
        myopicWinAction=None
        gameTreeWinAction=None
        for action in range(7):
            row=-1
            win=False
            row = self._47_CoinRowAfterAction(action)
            if row==-1:
                continue
            else:
                if row>0:
                    win=self._47_CanGameTreePlayerWin(row-1,action)
                if win==True:
#                    print("Action {0} is a losing action because in the next move GameTree Player will win".format(action))
                    losingAction.append(action)  #Losing action because in the next move GameTree Player will win
                else:
                    validAction.append(action)
            win=self._47_CanMyopicPlayerWin(row,action)
            if win==True:
#                print("Myopic can win : ",action,sep=" ")
                myopicWinAction=action
            win=self._47_CanGameTreePlayerWin(row,action)
            if win==True:
#                print("Game Tree can win : ",action,sep=" ")
                gameTreeWinAction=action
        return myopicWinAction, gameTreeWinAction, validAction, losingAction

    def _47_FindBestMyopicAction(self):
        myopicWinAction, gameTreeWinAction, validAction, losingAction = self._47_FindMyopicMoves()
#        print(myopicWinAction, gameTreeWinAction, validAction, losingAction)
        bestAction=None
        if myopicWinAction != None:
            bestAction = myopicWinAction
        elif gameTreeWinAction != None:
            bestAction = gameTreeWinAction
        elif len(validAction) > 0:
            bestAction = random.choice(validAction)
        elif len(losingAction) > 0:
            bestAction = random.choice(losingAction)
        assert bestAction!=None, "No actions are available for Myopic player"
        return bestAction

    def _47_TakeAction(self,action,player):
        row = self._47_CoinRowAfterAction(action)
        assert row!=-1, "Action {0} cannot be taken.".format(action)
        self._47_game[row][action] = player
        win = self._47_CanAPlayerWin(row,action,player)
        if win==True:
            self.winner=player
        print("Player {0} takes action {1}.".format(player, action))

    
    def MyopicPlayerAction(self):
        bestAction = self._47_FindBestMyopicAction()
        self._47_TakeAction(bestAction,player=1)
    
    def GameTreePlayerAction(self,action):
        assert action>=0 and action<=6, "Invalid game tree player action. Action out of range."
        self._47_TakeAction(action,player=2)

    def PrintGameState(self,state=None):
        game = self._47_game if state==None else state
        print(*[0, 1, 2, 3, 4, 5, 6],sep=" ")
        print(*['-','-','-','-','-','-','-'],sep=" ")
        for row in game:
            print(*row,sep=" ")
        print()
    
    def GetCurrentState(self):
        currentState = copy.deepcopy(self._47_game)
        return currentState
    
    def SetCurrentState(self,gameState):
        self._47_game = copy.deepcopy(gameState)


def main():
    pass

if __name__=='__main__':
    main()