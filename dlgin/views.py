from django.shortcuts import render
from dlgin.forms import UploadForm
from django.forms.formsets import formset_factory
from django.conf import settings
from librosa import feature, load, logamplitude
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
import numpy as np
import os


def index(request):
    UploadFormSet = formset_factory(UploadForm)

    if request.method == 'POST':
        upload_formset = UploadFormSet(request.POST, request.FILES)

        if upload_formset.is_valid():
            save_files(upload_formset)  # librosa can only read files from path, so we have to save all songs

            bytestring = calc_melgrams(request)

            labels, values = calc_genres(bytestring)

            print(labels, values)

    else:
        upload_formset = UploadFormSet()

    return render(request, 'dlgin/index.html', {
        'upload_formset': upload_formset
    })


def about(request):
    return render(request, 'dlgin/about.html')


def contact(request):
    return render(request, 'dlgin/contact.html')


def calc_genres(bytestring):
    response = requests.post('http://35.197.108.247:8080/predict', data=bytestring)

    labels = []
    values = []

    for key, value in response.json().items():
        labels.append(key)
        values.append(float(value))

    return labels, values


def calc_melgrams(request):
    slice_melgram = np.empty((128, 0))

    for filename in request.FILES:
        file_path = '{}/songs/{}'.format(settings.MEDIA_ROOT, request.FILES[filename].name)

        print("Calculating", request.FILES[filename].name)
        X, sr = load(file_path)
        melgram = logamplitude(feature.melspectrogram(y=X, sr=sr))

        slice_melgram = np.append(slice_melgram, melgram, axis=1)

        os.remove(file_path)  # delete song after done calculating

    print("")
    bytestring = slice_melgram.astype(np.float32).tobytes()

    return bytestring


def save_files(upload_formset):
    for form in upload_formset:
        form.save()
