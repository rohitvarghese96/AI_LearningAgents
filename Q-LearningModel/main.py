#import libraries
import numpy as np


#define training parameters
epsilon = 0.9
discountFactor = 0.9
learningRate = 0.9
totalEpisodes = 1000

#define environment
environment_rows = 10
envrironment_columns = 10

#define agent starting location
start_row = 9
start_column = 0

#initial Battery Points
initialBatteryPoints = 10000

#creating 3d numpr array holding the current Q-Values for each state and action pair
q_values = np.zeros((environment_rows, envrironment_columns, 4))

#define possible actions
actions = ['up', 'down', 'left', 'right']

rewards = np.full((environment_rows, envrironment_columns), -1)
rewards[9, 9] = 10000 #setting the reward for gold

#set rewards for metal sensing locations
#row0
rewards[0, 7] = -100
rewards[0, 8] = -100
rewards[0, 9] = -100


#row1
rewards[1, 2] = -100
rewards[1, 3] = -100
rewards[1, 4] = -100
rewards[1, 7] = -100
rewards[1, 8] = -100
rewards[1, 9] = -100


#row2
rewards[2, 2] = -100
rewards[2, 3] = -100
rewards[2, 4] = -100
rewards[2, 6] = -100
rewards[2, 7] = -100
rewards[2, 8] = -100
rewards[2, 9] = -100

#row3
rewards[3, 0] = -100
rewards[3, 1] = -100
rewards[3, 2] = -100
rewards[3, 3] = -100
rewards[3, 4] = -100
rewards[3, 6] = -100
rewards[3, 7] = -100
rewards[3, 8] = -100

#row4
rewards[4, 0] = -100
rewards[4, 1] = -100
rewards[4, 2] = -100
rewards[4, 3] = -100
rewards[4, 4] = -100
rewards[4, 6] = -100
rewards[4, 7] = -100
rewards[4, 8] = -100

#row5
rewards[5, 0] = -100
rewards[5, 1] = -100
rewards[5, 2] = -100
rewards[5, 3] = -100
rewards[5, 4] = -100

#row6
rewards[6, 2] = -100
rewards[6, 3] = -100
rewards[6, 4] = -100

#row7
rewards[7, 6] = -100
rewards[7, 7] = -100
rewards[7, 8] = -100

#row8
rewards[8, 1] = -100
rewards[8, 2] = -100
rewards[8, 3] = -100
rewards[8, 6] = -100
rewards[8, 7] = -100
rewards[8, 8] = -100

#row9
rewards[9, 1] = -100
rewards[9, 2] = -100
rewards[9, 3] = -100
rewards[9, 6] = -100
rewards[9, 7] = -100
rewards[9, 8] = -100


#function that determines if given location is a terminal state
def isTerminalState(currentRow, currentColumn):
    if rewards[currentRow, currentColumn] == -1:
        return False

    else:
        return True



#define next action based on epsilon greedy algorithm
def getNextAction(currentRow, currentColumn, epsilon):
    #if random value is less than epsilon, then produce next action based on the greedy-algorithm
    #else, return random action
    if np.random.random() < epsilon:
        return np.argmax(q_values[currentRow, currentColumn])
    else:
        return np.random.randint(4)



#function that will get next location based on chosen action
def getNextLocation(currentRow, currentColumn, actionIndex):
    newRow = currentRow
    newColumn = currentColumn

    if actions[actionIndex] == 'up' and currentRow < environment_rows - 1:
        newRow += 1

    elif actions[actionIndex] == 'down' and currentRow > 0:
        newRow -= 1

    elif actions[actionIndex] == 'left' and currentColumn > 0:
        newColumn -= 1

    elif actions[actionIndex] == 'right' and currentColumn < envrironment_columns - 1:
        newColumn += 1
    return newRow, newColumn



#function that will return the optimal path
def getoptimalPath(startRow, startColumn):
    batteryPoints = initialBatteryPoints

    if isTerminalState(startRow, startColumn):
        return []

    else:
        currentRow, currentColumn = startRow, startColumn
        optimalPath = []
        optimalPath.append([currentRow, currentColumn])

        #continue until gold is found
        while not isTerminalState(currentRow, currentColumn):
            #get next best action
            actionIndex = getNextAction(currentRow, currentColumn, 1.)

            #move to next location
            currentRow, currentColumn = getNextLocation(currentRow, currentColumn, actionIndex)

            reward = rewards[currentRow, currentColumn]
            batteryPoints = batteryPoints + reward

            #append the path
            optimalPath.append([currentRow, currentColumn])

        print('BatteryPoints:' + str(batteryPoints))
        return optimalPath



#Training the learning agent
for episode in range(totalEpisodes):
    currentRow = start_row
    currentColumn = start_column
    batteryPoints = initialBatteryPoints

    #continue taking actions while not in terminal state
    while not isTerminalState(currentRow, currentColumn) and batteryPoints>0:
        #choose which action to take
        actionIndex = getNextAction(currentRow, currentColumn, epsilon)

        #perform the next action
        oldRow, oldColumn = currentRow, currentColumn
        currentRow, currentColumn = getNextLocation(currentRow, currentColumn, actionIndex)

        #recieve the reward for moving to next state
        reward = rewards[currentRow, currentColumn]
        batteryPoints = batteryPoints + reward
        old_qValue = q_values[oldRow, oldColumn, actionIndex]
        temporalDifference = reward + (discountFactor * np.max(q_values[currentRow, currentColumn])) - old_qValue

        #update q-value for previous state and action pair
        new_qValue = old_qValue + (learningRate * temporalDifference)
        q_values[oldRow, oldColumn, actionIndex] = new_qValue


print(getoptimalPath(start_row, start_column))
