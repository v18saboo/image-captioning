from nltk.translate.bleu_score import sentence_bleu
print 1+2
hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
#there may be several references
BLEUscore = sentence_bleu([reference], hypothesis, weights=[0.333,0.333,0.333])
print BLEUscore