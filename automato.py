import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn import neural_network as nn


'''
Grupo:
    Carlos Gabriel
    Isaque Almeida
    Luis Fernando
/-----/

MAN WOLF GOAT CABBAGE
MWGC
0000
1010
0010
1011
0001
1101
0101
1111

Q0  >> g >> Q1
Q1  >> f >> Q2
Q2  >> c >> Q4
Q4  >> g >> Q8
Q8  >> a >> Q12
Q12 >> f >> Q16
Q16 >> g >> Q18

Expression: GFCGAFG
'''
X_I = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    ])

X_test = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

y_O = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    ])

def compare(a, b):
    k = [sum(a[i]) for i in range(np.shape(a)[0])]
    z = [sum(b[i]) for i in range(np.shape(b)[0])]

    result = np.logical_xor(k, z)

    if any(result):
        return False
    else:
        return True

l_rate = 1e-5
neural = nn.MLPClassifier(
    hidden_layer_sizes=(60,),
    max_iter=256,
    solver='lbfgs',
    alpha=1e-4,
    verbose=10,
    random_state=1,
    learning_rate_init=l_rate
    )

print('\nTrainning...')
neural.fit(X_I, y_O)
print('\nTesting...')
y = neural.predict(X_I)
print(f"Test Score: {neural.score(X_I, y_O)}")

print(f"""The result of test is the same as expected?
{'R: Yes.' if compare(y, y_O) else 'No, the result is incorrect.'}
""")

print("Checking expression...")
j = neural.predict(X_test)
print(j[0])
for i in range(2,7):
    j = neural.predict(j)
    print(j[0])
