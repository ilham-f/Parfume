from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import AuthenticationForm
from .forms import PerfumePreferenceForm
from django.db.models import Count, Avg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os
from django.conf import settings
import pandas as pd

def home(request):
    parfumes = [
        { "name": "Dior Sauvage", "price": 120 },
        { "name": "Chanel Bleu de Chanel", "price": 135 },
        { "name": "Yves Saint Laurent La Nuit De L'Homme", "price": 110 },
        { "name": "Creed Aventus", "price": 350 },
        { "name": "Tom Ford Oud Wood", "price": 250 },
        { "name": "Versace Eros", "price": 95 },
        { "name": "Armani Acqua di Gio", "price": 105 },
        { "name": "Jean Paul Gaultier Le Male", "price": 98 },
    ]

    context = {
        "parfumes": parfumes
    }
    return render(request, 'home.html', context)

def about(request):
    return render(request, 'about.html')

def recommendations(request):
    if request.method == "POST":
        form = PerfumePreferenceForm(request.POST)
        
        if form.is_valid():
            scent = form.cleaned_data.get("scent", "")
            base_note = form.cleaned_data.get("base_note", "")
            middle_note = form.cleaned_data.get("middle_note", "")
            ml = form.cleaned_data.get("ml") or 0
            concentration = form.cleaned_data.get("concentration", "")
            price = form.cleaned_data.get("price") or 0
            
            print(scent)
            print(base_note)
            print(middle_note)
            print(ml)
            print(concentration)
            print(price)

        # Load CSV
        file_csv = os.path.join(settings.BASE_DIR, 'static', 'data', 'data_parfume.csv')
        data = pd.read_csv(file_csv, delimiter=';')

        # Process Cleaning Data
        data = data.drop_duplicates()
        data = data.dropna(subset=['scents', 'base_note', 'middle_note', 'name', 'brand', 'price', 'ml', 'concentration'])
        data['price'] = pd.to_numeric(data['price'], errors='coerce')
        data['ml'] = pd.to_numeric(data['ml'], errors='coerce')
        data = data.dropna(subset=['price', 'ml'])

        # Membersihkan teks dan mengubah ke huruf kecil
        data['scents'] = data['scents'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
        data['base_note'] = data['base_note'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
        data['middle_note'] = data['middle_note'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
        data['concentration'] = data['concentration'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()

        # Kombinasi 'scents', 'base_note', dan 'middle_note'
        data['combined_notes'] = data['scents'] + ' ' + data['base_note'] + ' ' + data['middle_note']

        # Menggunakan TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data['combined_notes'])

        # Clustering
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['cluster'] = kmeans.fit_predict(X)

        # Fungsi rekomendasi
        def recommend_based_on_input(input_scent, input_base_note, input_middle_note, input_ml, input_concentration, input_price):
            combined_input = (input_scent or '') + ' ' + (input_base_note or '') + ' ' + (input_middle_note or '')
            input_vector = vectorizer.transform([combined_input])

            # Prediksi cluster
            input_cluster = kmeans.predict(input_vector)[0]

            # Ambil parfum dalam cluster yang sama
            cluster_parfumes = data[data['cluster'] == input_cluster]

            # Hitung cosine similarity
            similarity_scores = cosine_similarity(input_vector, vectorizer.transform(cluster_parfumes['combined_notes']))

            # Urutkan berdasarkan similarity
            similar_parfum_indices = similarity_scores.argsort()[0][::-1]
            filtered_parfumes = cluster_parfumes.iloc[similar_parfum_indices]

            # Filter berdasarkan input pengguna
            filtered_parfumes = filtered_parfumes[
                (filtered_parfumes['ml'] >= input_ml - 10) & 
                (filtered_parfumes['ml'] <= input_ml + 10) & 
                (filtered_parfumes['concentration'].str.contains(input_concentration, case=False)) & 
                (filtered_parfumes['price'] <= input_price)
            ]

            # Ambil 5 rekomendasi teratas
            recommended_parfumes = filtered_parfumes.head(5)

            return recommended_parfumes[['brand', 'name', 'scents', 'base_note', 'middle_note', 'ml', 'concentration', 'price']].to_dict(orient='records')

        # Menjalankan rekomendasi
        recommended_parfumes = recommend_based_on_input(scent, base_note, middle_note, ml, concentration, price)

        context = {
            "parfumes": recommended_parfumes
        }

        return render(request, 'result.html', context)

def recommendation_form(request):
    if request.method == 'POST':
            form = PerfumePreferenceForm(request.POST)
            if form.is_valid():
                user = form.save()
                auth_login(request, user)
                return redirect('home')
            else:
                print(form.errors)
    else:
        form = PerfumePreferenceForm()
        
        return render(request, 'recommendation_form.html', {'form': form})