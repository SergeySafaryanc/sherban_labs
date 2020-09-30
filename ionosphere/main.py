from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from itertools import product

data = pd.read_csv('./ionosphere.data')
gen_comb = lambda **kwargs: list(product(*kwargs.values()))


def to_int(x):
    return {
        'b': np.array(0),
        'g': np.array(1)
    }.get(x)


# X = data.iloc[:, :-1].to_numpy()
# y = [to_int(i) for i in data.iloc[:, -1].values]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=14)
#
# res = {}
# for i in gen_comb(n_neighbors=range(1, len(X_train), 5)):
#     knn = KNeighborsClassifier(n_neighbors=i[0], n_jobs=-1)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     print(f"Конфигурация[n_neighbors={i[0]}] - {np.mean(y_pred == y_test)}")
#     res.update({i[0]: np.mean(y_pred == y_test)})
#
# sns.displot(data=res, x="total_bill", col="time", kde=True)

# for k, v in res.items():
#     print(k, v)
