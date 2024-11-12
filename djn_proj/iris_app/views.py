from django.shortcuts import render
from django.http.response import HttpResponse
import joblib
# Create your views here.
def index(request):
    print('index 뷰 동작 확인')
    return render(request, 'iris_app/index.html')


def modelloadtest(request):
    clf = joblib.load('ML_model/svc_model.pkl')
    pre = clf.predict([[5.1, 4.5, 2.4, 2.2]])
    
    iris_types = ['setosa', 'versicolor', 'virginica']
    output = iris_types[int(pre[0])]
    print("pre : \n", output)
    
    return HttpResponse(output)

def predict(request):
    sl = float(request.POST['sl'])         # 꽃받침 길이 (sepal_length)
    sw = float(request.POST['sw'])  # 꽃받침 너비 (sepal width)
    pl = float(request.POST['pl'])  # 꽃잎 길이 (petal length)
    pw = float(request.POST['pw'])  # 꽃잎 너비 (petal width)
    
    clf = joblib.load('ML_model/svc_model.pkl')
    prediction = clf.predict([[sl, sw, pl, pw]])


    iris_types = ['setosa', 'versicolor', 'virginica']
    result = iris_types[int(prediction[0])]

    context = {
        'sl' : sl,
        'sw' : sw,
        'pl' : pl,
        'pw' : pw,
        'result' : result,
        'iris_img': result+'.png'
    }
    # for commit
    return render(request, 'iris_app/result.html', context)