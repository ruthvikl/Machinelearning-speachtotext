from django.shortcuts import render
#Tensorflow/Keras
import tensorflow as tf
# import speech_recognition as sr
import tensorflow_hub as hub
import tensorflow_text as text
# from official.nlp import optimization# Create your views here.
text_recog = []

def home(request):
    return render(request, 'index.html')

def prediction(request):
    result = request.GET.get('result', None)
    print(result)
    model = tf.keras.models.load_model('model_py/new_IMDb_bert', compile=False)
    text_recog.append(result)
    print(text_recog)
    prediction = tf.sigmoid(model(tf.constant(text_recog)))
    text_recog.clear()
    prediction = f'{prediction[0][0]*10:.3f}'
    print(text_recog)
    print(prediction)
    return render(request, 'index.html')
