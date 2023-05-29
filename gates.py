import random
import math

#or gate
train_data = [
    [0,0,0],
    [1,0,1],
    [0,1,1],
    [1,1,1],
]


data_length = len(train_data)


#map any range to -1,1
def sigmoid(x):
    return 1 / (1 + math.exp(-x))



def findCost(weight1, weight2, bias):
    result = 0

    for values in train_data:
        x1 = values[0]
        x2 = values[1]
        y = sigmoid(x1*weight1 + x2*weight2 + bias)

        distance = y - values[2]
        #square is a way to keep numbers positive, while amplifying values at the same time
        result += distance * distance 

        #print(f"our prediction: {y} expected: {values[1]}")

    cost = result / data_length
    return cost




def main():
    weight1 = random.random()
    weight2 = random.random()
    bias = random.random()

    eps = 1e-1 
    rate = 1e-1 

    for i in range(100000):
        cost = findCost(weight1, weight2, bias)
        #print(f"weight1: {weight1} | weight2: {weight2} | bias: {bias} | cost: {cost}")
        dist_weight1 = (findCost(weight1 + eps, weight2, bias) - cost)/eps
        dist_weight2 = (findCost(weight1, weight2 + eps, bias) - cost)/eps
        dist_bias = (findCost(weight1, weight2, bias + eps) - cost)/eps
        weight1 -= rate*dist_weight1
        weight2 -= rate*dist_weight2
        bias -= rate*dist_bias
        #print(cost)

    print(f"result cost: {cost}")

main()