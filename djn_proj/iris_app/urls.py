from django.urls import path
from iris_app import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'iris_app'

urlpatterns = [
    path("",views.index, name='index'),
    path("test1/", views.modelloadtest, name='test1'),
    path("result/", views.predict, name='result'),
] + static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)



