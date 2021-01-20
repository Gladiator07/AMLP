from custom_metrics import accuracy, accuracy_v2
from custom_metrics import precision, recall,f1
from sklearn.metrics import accuracy_score, f1_score

# checking both the accuracy function created with sci-kit learn implementation


l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]

a1 = accuracy(l1, l2)
a2 = accuracy_v2(l1, l2)
sk = accuracy_score(l1, l2)
p = precision(l1, l2)
r = recall(l1, l2)
f1_c = f1(l1, l2)
f1_s = f1_score(l1, l2)

print(f"Accuracy of the first function: {a1}")
print(f"Accuracy of the second(using tp,fp,fn,tn) function: {a2}")
print(f"Accuracy of sklearn's implementation: {sk}") 

print(f"Precision: {p}")
print(f"Recall: {r}")
print(f"f1 (custom metric): {f1_c}")
print(f"f1 (sklearn metric): {f1_s}")