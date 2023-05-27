import random
import time

training_data = [
    [0, 0], 
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8]
]

#u a "random" guess number
#return an indication of how well this model works(0 is perfect), this is more or less an average calculation
def findCost(u, bias):
    result = 0

    for values in training_data:
        x = values[0]
        y = x*u + bias

        distance = y - values[1]
        #square is a way to keep numbers positive, while amplifying values at the same time
        result += distance * distance 

        #print(f"our prediction: {y} expected: {values[1]}")

    cost = result / len(training_data)
    return cost


def main():
    #random number to predict(this is our prediction)
    w = random.random() * 10.0
    bias = random.random() *  2.0

    #'magical' values
    eps = 1e-3 #epsilon, is some kind of tunning constant
    rate = 1e-3 #learning rate, keeps our values small 

    #iterating and fine tuning our values
    for i in range(1000):
        #calculating a derivative, basically find if we are moving towards or further from out goal(cost = 0)
        w_diff = (findCost(w + eps, bias) - findCost(w, bias)) / eps
        bias_diff = (findCost(w, bias + eps) - findCost(w, bias)) / eps
        #and then start moving there by fine tuning our value
        w -= rate*w_diff
        bias -= rate*bias_diff
        #and we calculate again
        cost = findCost(w, bias)
        print(f"cost: {cost} | w: {w} | bias: {bias}")

    print("------------------------")
    print("prediction:", w)

main()