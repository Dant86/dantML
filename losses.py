import numpy as np
import math

def crossentropy(logits, labels):
	sum = 0
	for i in range(len(logits)):
		curr = 0
		curr += labels[i]*math.log(logits[i])
		curr += (1-labels[i])*math.log(1-logits[i])
		sum += curr
	return -sum / len(logits)