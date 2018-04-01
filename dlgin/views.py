from django.shortcuts import render
from dlgin.forms import UploadForm
from django.forms.formsets import formset_factory
from django.conf import settings
import librosa
from librosa import feature, logamplitude
from scipy.misc import imsave
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

            render_graphic(labels, values)

    else:
        upload_formset = UploadFormSet()
        img = np.zeros([100, 100, 3], dtype=np.uint8)
        img.fill(255)
        imsave(settings.BASE_DIR + '/dlgin/static/img/figure.png', img)

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

    return labels, np.array(values)


def calc_melgrams(request):
    slice_melgram = np.empty((128, 0))

    for filename in request.FILES:
        file_path = '{}/songs/{}'.format(settings.MEDIA_ROOT, request.FILES[filename].name)

        print("Calculating", request.FILES[filename].name)
        X, sr = librosa.load(file_path)
        melgram = logamplitude(feature.melspectrogram(y=X, sr=sr))

        slice_melgram = np.append(slice_melgram, melgram, axis=1)

        os.remove(file_path)  # delete song after done calculating

    bytestring = slice_melgram.astype(np.float32).tobytes()

    return bytestring


def save_files(upload_formset):
    for form in upload_formset:
        form.save()


def render_graphic(labels, values):
    fig = plt.figure()
    plt.gcf().subplots_adjust(bottom=0.30)  # make room for labels
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
              '#d2f53c', '#008080', '#e6beff', '#aa6e28', '#800000', '#aaffc3', '#000080', '#ffd8b1']

    if values.ndim == 1:  # if only one slice to display
        ypos = np.arange(len(labels))

        plt.bar(ypos, values, align='edge', color=colors)

        plt.xticks(ypos, labels, rotation=70)
        plt.xlabel('Genre')

    else:
        ax = Axes3D(fig)

        lx = len(values[0])
        ly = len(values[:, 0])

        xpos = np.arange(0, lx, 1)
        ypos = np.arange(0, ly, 1)
        xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)

        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(lx * ly)

        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = values.flatten()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)
        ax.w_xaxis.set_ticklabels(labels)
        ax.set_xlabel('Genre')

    plt.savefig(settings.BASE_DIR + '/dlgin/static/img/figure.png')
