import matplotlib
matplotlib.use("Agg")
from django.shortcuts import render, redirect
from django.views.generic import TemplateView
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

from .forms import MusicForm
from .models import Music

import pandas_gbq
from google.oauth2 import service_account

from mer.main import main
from mer.learn_kde_audio import plot_pdf
import os
from pathlib import Path

import io
import matplotlib.pyplot as plt
import numpy as np

import random
from matplotlib import pylab
from pylab import *

import numpy as np

import PIL, PIL.Image
from io import StringIO, BytesIO
# Create your views here.

credentials = service_account.Credentials.from_service_account_file(r'/Users/zhonglingjiang/Desktop/big-data-analytics-526ab9f8025b.json')


class Home(TemplateView):
    template_name = 'home.html'
    
def index(request):
    return HttpResponse("<h1>Welcome to music recommender app<h1>")

def base(request):
    return render(request, 'base.html')

def recommend(request):
    context = {}
    #uploaded_file = None
    global uploaded_file
    global url  ## FIX THIS 
    global pdf
    if request.method == 'POST':
        if '_upload' in request.POST:
            uploaded_file = request.FILES['document']
            # print(uploaded_file.size)
            fs = FileSystemStorage()
            name = fs.save(uploaded_file.name, uploaded_file)
            url = fs.url(name)
            context['url'] = url
            return render(request, 'recommend.html', context)
        elif '_recommend' in request.POST:
            # recommend_music(request)
            pandas_gbq.context.credentials = credentials
            pandas_gbq.context.project = "big-data-analytics-252415"
            SQL_1 = """
                SELECT *
                FROM
                `big-data-analytics-252415.music_emotion_recognition.music_features` 
            """

            SQL_2 = """
                SELECT * 
                FROM 
                `big-data-analytics-252415.music_emotion_recognition.pdf`
            """
            df_1 = pandas_gbq.read_gbq(SQL_1).set_index('song_id')
            df_2 = pandas_gbq.read_gbq(SQL_2).set_index('song_id')

            if uploaded_file is None:
                print('There is no file uploaded..')
            else:
                # print(uploaded_file.size)
                # use main() API to retrieve a list of recommended song ids.
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                song_path = Path(base_dir).parent / url[1:]
                train_audio = df_1
                train_pdfs = df_2
                num_recs = 5
                recs_id_kl_dict, pdf = main(song_path, train_audio, train_pdfs, num_rec=num_recs)
                recs_id = [i for i in recs_id_kl_dict]
                recs_kl_score = [recs_id_kl_dict[i] for i in recs_id_kl_dict]
                
                # recs_id is a list of song id
                # use these to find song names, artists, genre information 
                placeholders = ', '.join(str(i) for i in recs_id)
                SQL_3 = """
                    SELECT song_id, file_name, Artist, Song_title, Genre 
                    FROM 
                    `big-data-analytics-252415.music_emotion_recognition.songs_info`
                    WHERE song_id IN (%s);
                """ % placeholders
                df_3 = pandas_gbq.read_gbq(SQL_3)
                df_3['score'] = df_3['song_id'].apply(lambda x: recs_id_kl_dict[x])
                df_3 = df_3.sort_values(by=['score'], ascending=False)

                song_loc_prefix = Path('/media/music/clips_45seconds/')
                rec_songs = [{
                    'song_id': row['song_id'],
                    'Artist': row['Artist'],
                    'Song_title': row['Song_title'],
                    'Genre': row['Genre'], 
                    'Score': np.round(row['score'] * 1000000, 3),
                    'music_url': song_loc_prefix / row['file_name']
                } for (index, row) in df_3.iterrows()]

            recs_result = {'text': rec_songs} ## FIX THIS
            
            return render(request, 'display_recommendation.html', recs_result)
    else:         
        return render(request, 'recommend.html', context)

def music_list(request):
    music = Music.objects.all()
    
    return render(request, 'music_list.html', {
        'music': music
    })

def upload_music(request): 
    if request.method == 'POST':
        form = MusicForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('music_list')
    else:
        form = MusicForm()
    return render(request, 'upload_music.html', {
        'form': form
    })

def showimage(request):

    """ Plots a heatmap of the pdf in the VA space"""
    pdf_square = np.flipud(pdf.reshape(16, 16))
    val = np.round(np.linspace(-1, 1, 16), 2)
    ar = np.round(np.linspace(-1, 1, 16), 2)

    fig, ax = plt.subplots()
    # im = ax.imshow(pdf_square)

    # # We want to show all ticks...
    ax.set_xticks(np.arange(len(val)))
    ax.set_yticks(np.arange(len(ar)))
    # # ... and label them with the respective list entries
    ax.set_xticklabels(val, rotation=45)
    ax.set_yticklabels(ar)

    ax.set_title("Emotion Map of Uploaded Song")
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')

    fig.tight_layout()

    pcolor(pdf_square)
    colorbar()
 
    # Store image in a string buffer
    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()
 
    # Send buffer in a http response the the browser with the mime type image/png set
    return HttpResponse(buffer.getvalue(), content_type="image/png")


