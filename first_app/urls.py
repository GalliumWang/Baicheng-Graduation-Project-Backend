from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_view, name='predict'),
    path('airplane_id/', views.airplane_id_view, name='airplane_id'),
    # path('feature_importance/', views.feature_importance_view,
    #      name='feature_importance'),
]
