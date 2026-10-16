from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split

X = [
    [181, 80, 44], [178, 75, 43], [182, 82, 44], [188, 90, 46],
    [175, 74, 42], [192, 95, 47], [180, 78, 43], [185, 85, 45],
    [173, 70, 42], [179, 77, 43], [183, 83, 44], [186, 88, 45],
    [176, 73, 42], [190, 92, 46], [174, 71, 41], [184, 84, 44],
    [168, 62, 40], [165, 58, 39], [170, 64, 40], [167, 61, 39],  
    
    [162, 55, 37], [165, 58, 38], [158, 52, 36], [170, 63, 39],
    [160, 54, 37], [168, 62, 39], [155, 50, 36], [163, 56, 38],
    [167, 60, 38], [159, 53, 37], [164, 57, 38], [169, 64, 39],
    [157, 51, 36], [161, 55, 37], [166, 59, 38], [171, 65, 40],
    [175, 72, 41], [177, 74, 42], [173, 69, 41], [174, 70, 41]  
]

Y = (['male'] * 20) + (['female'] * 20)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=42)

dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train,y_train)

reg = linear_model.LogisticRegression()
reg.fit(X_train, y_train)

gd = ensemble.GradientBoostingClassifier()
gd = gd.fit(X_train, y_train)

models = [dt, reg, gd]
for model in models:
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(model, ":" , score)