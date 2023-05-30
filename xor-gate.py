import random
import math
import time

random.seed(time.time())
#or gate
or_data = [
    [0,0,0],
    [1,0,1],
    [0,1,1],
    [1,1,1],
]
#and gate
and_data = [
    [0,0,0],
    [1,0,0],
    [0,1,0],
    [1,1,1],
]
#nand gate
nand_data = [
    [0,0,1],
    [1,0,1],
    [0,1,0],
    [1,1,0],
]
#xor gate
xor_data = [
    [0,0,0],
    [1,0,1],
    [0,1,1],
    [1,1,0],
]

train_data = or_data

#training a 3 neuron network
#for example: feeding 'or' and 'nand' results to an 'and' will give us a 'xor'(this is NOT what will happen every time)

class Xor:
    def __init__(self):
        self.or_x1 = random.random()
        self.or_x2 = random.random()
        self.or_b = random.random()
        self.nand_x1 = random.random()
        self.nand_x2 = random.random()
        self.nand_b = random.random()
        self.and_x1 = random.random()
        self.and_x2 = random.random()
        self.and_b = random.random()
        
    
    def forward(self,x1, x2):
        or_res = sigmoid(x1*self.or_x1 + x2*self.or_x2 + self.or_b)
        nand_res = sigmoid(x1*self.nand_x1 + x2*self.nand_x2 + self.nand_b)
        
        return sigmoid(or_res*self.and_x1 + nand_res*self.and_x2 + self.and_b)


    def print_data(self):
        print(f"or_x1:{self.or_x1} | or_x2:{self.or_x2} | or_b:{self.or_b}")
        print(f"nand_x1:{self.nand_x1} | nand_x2:{self.nand_x2} | nand_b:{self.nand_b}")
        print(f"and_x1:{self.and_x1} | and_x2:{self.and_x2} | and_b:{self.and_b}")

    def finite_diff(self,eps):
        next_xor = Xor()

        original_cost = findCost(self)

        saved = self.or_x1
        self.or_x1 += eps 
        next_xor.or_x1 = (findCost(self) - original_cost)/eps
        self.or_x1 = saved
        
        saved = self.or_x2
        self.or_x2 += eps 
        next_xor.or_x2 = (findCost(self) - original_cost)/eps
        self.or_x2 = saved
        
        saved = self.or_b
        self.or_b += eps 
        next_xor.or_b = (findCost(self) - original_cost)/eps
        self.or_b = saved
        
        saved = self.nand_x1
        self.nand_x1 += eps 
        next_xor.nand_x1 = (findCost(self) - original_cost)/eps
        self.nand_x1 = saved

        saved = self.nand_x2
        self.nand_x2 += eps 
        next_xor.nand_x2 = (findCost(self) - original_cost)/eps
        self.nand_x2 = saved
        
        saved = self.nand_b
        self.nand_b += eps 
        next_xor.nand_b = (findCost(self) - original_cost)/eps
        self.nand_b = saved
        
        saved = self.and_x1
        self.and_x1 += eps 
        next_xor.and_x1 = (findCost(self) - original_cost)/eps
        self.and_x1 = saved

        saved = self.and_x2
        self.and_x2 += eps 
        next_xor.and_x2 = (findCost(self) - original_cost)/eps
        self.and_x2 = saved
        
        saved = self.and_b
        self.and_b += eps 
        next_xor.and_b = (findCost(self) - original_cost)/eps
        self.and_b = saved
        
        return next_xor
     
    def learn(self, new_xor, rate):
        self.or_x1 -= rate*new_xor.or_x1
        self.or_x2 -= rate*new_xor.or_x1
        self.or_b -= rate*new_xor.or_x1
        self.nand_x1 -= rate*new_xor.nand_x1
        self.nand_x2 -= rate*new_xor.nand_x2
        self.nand_b -= rate*new_xor.nand_b
        self.and_x1 -= rate*new_xor.and_x1
        self.and_x2 -= rate*new_xor.and_x2
        self.and_b -= rate*new_xor.and_b

        return self

#map any range to -1,1
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def findCost(xor_trainer):
    result = 0

    for values in train_data:
        x1 = values[0]
        x2 = values[1]
        y = xor_trainer.forward(x1, x2)

        distance = y - values[2]
        #square is a way to keep numbers positive, while amplifying values at the same time
        result += distance * distance 

        #print(f"our prediction: {y} expected: {values[1]}")

    cost = result / len(train_data)
    return cost



def main():
    xor_trnr = Xor()

    eps = 1e-1
    rate = 1e-1

    for i in range(100000):
        xor_new = xor_trnr.finite_diff(eps)

        xor_trnr.learn(xor_new, rate)
        
    print(f"cost: {findCost(xor_trnr)}")

    for i in range(2):
        for j in range(2):
            print(f"{i} ^ {j} = {xor_trnr.forward(i, j)}")
main()