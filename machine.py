import random
import time

training_data = [
    [0, 0], 
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8]
]
# searching for the unknown u number
#y = x * u


#u a "random" guess number
#return an indication of how well this model works(0 is perfect), this is more or less an average calculation
def findCost(u):
    result = 0

    for values in training_data:
        x = values[0]
        y = x*u

        distance = y - values[1]
        #square is a way to keep numbers positive, while amplifying values at the same time
        result += distance * distance 

        #print(f"our prediction: {y} expected: {values[1]}")

    cost = result / len(training_data)
    return cost

def main():
    #random number to predict(this is our prediction)
    w = random.random() * 10.0
    
    eps = 1e-3 #epsilon, is some kind of tunning constant
    rate = 1e-3 #another one

    #iterating and fine tuning our values
    for i in range(1000):
        #distance of cost, this formula calculates error change between iterations
        dcost = (findCost(w + eps) - findCost(w)) / eps
        #and here we do some kind of 'fine tuning' of our value
        w -= rate*dcost 
        #and we calculate again
        cost = findCost(w)
        print(f"cost: {cost} | w: {w}")

    print("prediction:", w)

main()