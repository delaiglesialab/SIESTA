# Standard Library Imports
import os
import subprocess
import hmac
import hashlib

# Third-Party Imports
import pandas as pd

# Django Imports
from django.http import HttpResponse, JsonResponse, HttpResponseForbidden
from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

# Custom Imports
from siesta_app.forms import FeatureExtractionForm
from siesta_app.tasks import download_feat, train_new_model, sleep_prediction
from celery.result import AsyncResult

MEDIA = f'{settings.MEDIA_ROOT}'
CHUNK_SIZE = 1024 * 1024 * 50

# Main URLS

def index(request):
    if request.META.get("HTTP_HX_REQUEST") != 'true':
        return render(request, 'siesta_app/index_full.html')

    if not request.session.session_key:
        request.session.save()

    return render(request, "siesta_app/index.html")


def extract_features(request):
    if not request.session.session_key:
        request.session.save()
        request.session.modified = True

    if request.method == "GET" and request.META.get("HTTP_HX_REQUEST") == 'true':
        form_data = request.session.get('form_data')
        form = FeatureExtractionForm(initial=form_data)
        return render(request, 'siesta_app/extract_features.html', {'form': form})

    if request.method == "POST" and request.POST.get('ajax_update') == 'true':
        form = FeatureExtractionForm(request.POST, request.FILES)
        if form.is_valid():
            form_data = form.cleaned_data
            request.session['form_data'] = form_data
            request.session['fs'] = form_data['fs']
            request.session['epoch'] = form_data['epoch']
            request.session['ECoG1_chan'] = form_data['ECoG1_chan']
            request.session['ECoG2_chan'] = form_data['ECoG2_chan']
            request.session['EMG_chan'] = form_data['EMG_chan']
            request.session.modified = True
            request.session.save()
            return JsonResponse({"message": "Parameters updated successfully!"})
        return JsonResponse({"error": "Invalid form data"}, status=400)

    if request.headers.get('x-requested-with') == 'XMLHttpRequest' and request.method == "POST":
        blobName = request.POST.get('fileName', None)

        if request.session.get('ECoG2_chan') is None:
            extracted_data = download_feat.delay(
                request.session.session_key, blobName,
                request.session.get('fs'), request.session.get('epoch'),
                request.session.get('ECoG1_chan') - 1,
                request.session.get('EMG_chan') - 1
            )
        else:
            extracted_data = download_feat.delay(
                request.session.session_key, blobName,
                request.session.get('fs'), request.session.get('epoch'),
                request.session.get('ECoG1_chan') - 1,
                request.session.get('EMG_chan') - 1,
                request.session.get('ECoG2_chan') - 1
            )
        return JsonResponse({"task_id": extracted_data.task_id}, status=202)

    form_data = request.session.get('form_data')
    form = FeatureExtractionForm(initial=form_data)
    return render(request, "siesta_app/extract_features_full.html", {'form': form})


def score_data(request):
    if request.META.get("HTTP_HX_REQUEST") == 'true':
        return render(request, 'siesta_app/score_data.html')

    if request.method == 'POST':
        modelType = request.POST.get('modelType')
        modelName = request.POST.get('modelName')
        fileName = request.POST.get('fileName')

        if fileName:
            scored_data = sleep_prediction.delay(request.session.session_key, fileName, modelType, modelName)

            return JsonResponse({"task_id": scored_data.task_id}, status=202)

        else:
            return JsonResponse({"error": "No file uploaded."}, status=400)

    return render(request, "siesta_app/score_data_full.html")


def merge_data(request):
    if request.META.get("HTTP_HX_REQUEST") == 'true':
        return render(request, 'siesta_app/merge_data.html')

    if request.method == 'POST':
        data_file = request.FILES.get('data_file')
        score_file = request.FILES.get('score_file')

        if not data_file or not score_file:
            return JsonResponse({'error': 'Both files must be provided.'}, status=400)

        try:
            merged_data = pd.read_csv(data_file, header=0, parse_dates=[-2])
            scores = pd.read_csv(score_file, header=None).iloc[:, 0].tolist()

            if len(merged_data) != len(scores):
                return JsonResponse({"error": "File lengths do not match."}, status=400)
            else: merged_data['score'] = scores

            if all(pd.to_numeric(merged_data['score'], errors='coerce').notna()):
                merged_data['score'] = merged_data['score'].map({1: 'WAKE', 2: 'NREM', 3: 'REM'})

            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename=merged_data.csv'
            merged_data.to_csv(response, index=False)

            return response
        except Exception as e:
            error_message = 'An error occurred during when merging: {}'.format(str(e))
            return render(request, "siesta_app/merge_data_full.html", {'error_message': error_message})

    return render(request, "siesta_app/merge_data_full.html")


def fit_model(request):
    if request.META.get("HTTP_HX_REQUEST") == 'true':
        return render(request, 'siesta_app/fit_model.html', {})

    if not request.session.session_key:
        request.session.save()

    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        request.session['cache_new_model'] = request.POST.get('cache_model', 'false') == 'true'
        uploaded_files = request.POST.getlist('uploaded_files[]')
        include_training_data = request.POST.get('include_training_data', 'false') == 'true'
        training_chan = request.POST.get('training_chan')

        new_model = train_new_model.delay(request.session.session_key, uploaded_files, include_training_data, training_chan)

        return JsonResponse({"task_id": new_model.id})

    return render(request, "siesta_app/fit_model_full.html", {})


# Helper URLS

def validate_task_state(request):
    task_id = request.GET.get('task_id')
    task_state = AsyncResult(task_id).state if task_id else "FAILURE"
    return JsonResponse({'state': task_state}, status=202)


def upload_file(request):
    if not request.session.session_key:
        request.session.save()

    if not os.path.join(MEDIA, request.session.session_key):
        os.makedirs(os.path.join(MEDIA, request.session.session_key))

    if request.method == 'POST' and request.FILES.get('file'):
        file_name = request.POST.get('fileName')
        chunk_index = int(request.POST.get('chunkIndex'))
        total_chunks = int(request.POST.get('totalChunks'))
        file_path = os.path.join(MEDIA, request.session.session_key, file_name)

        try:
            with open(file_path, 'ab') as destination:
                for chunk in request.FILES['file'].chunks():
                    destination.write(chunk)

            if chunk_index + 1 == total_chunks:
                response_data = {'blobName': file_name, 'status': 'success'}
                return JsonResponse(response_data)

            return JsonResponse({'message': 'Chunk uploaded successfully.'})

        except OSError as e:
            logger.error(f"Error writing file {file_name}: {e}")
            return JsonResponse({'error': 'File upload failed.'}, status=500)

    return JsonResponse({'error': 'Invalid request method or file not found.'}, status=400)


def download_csv(request):
    task_id = request.GET.get('task_id')
    filename = request.GET.get('filename')
    save_to_server = request.GET.get('save_to_server') == 'true'

    if task_id:
        data = pd.read_json(AsyncResult(task_id).get())

        if save_to_server:
            data.to_csv(os.path.join(MEDIA, request.session.session_key, filename), index=False)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename={filename}'
        data.to_csv(response, index=False)

        return response
    else:
        return HttpResponse('No job id given.')


def download_model(request):
    task_id = request.GET.get('task_id')

    if task_id:
        model = AsyncResult(task_id).get()
        cache_new_model = request.session.get('cache_new_model', False)

        file_path = os.path.join(MEDIA, request.session.session_key, model)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                file_content = file.read()

            response = HttpResponse(file_content, content_type='text/plain')
            response['Content-Disposition'] = 'attachment; filename=new_model.file'
            response['X-Cache-New-Model'] = str(cache_new_model).lower()
            response['X-Model-Name'] = model

            if not cache_new_model:
                os.remove(file_path)

            return response

        else:
            return HttpResponse('File not found.', status=404)

    else:
        return HttpResponse('No job id given.', status=400)


def check_file_exists(request):
    if not request.session.session_key:
        request.session.save()

    if not os.path.exists(os.path.join(settings.MEDIA_ROOT, request.session.session_key)):
        os.makedirs(os.path.join(settings.MEDIA_ROOT, request.session.session_key))

    file_name = request.GET.get("fileName")
    file_path = os.path.join(MEDIA, request.session.session_key, file_name)

    if os.path.exists(file_path):

        if request.GET.get('delete') == 'true':
            os.remove(file_path)

            return JsonResponse({"exists": False})

        return JsonResponse({"exists": True})

    return JsonResponse({"exists": False})


def check_server_files(request):
    if not request.session.session_key:
        request.session.save()

    if not os.path.exists(os.path.join(settings.MEDIA_ROOT, request.session.session_key)):
        os.makedirs(os.path.join(settings.MEDIA_ROOT, request.session.session_key))

    files = [f for f in os.listdir(os.path.join(MEDIA, request.session.session_key)) if 'features' in f]
    return JsonResponse({'files': files})


@csrf_exempt
def github_webhook(request):
    secret = f'{settings.GIT_KEY}'
    header_signature = request.META.get('HTTP_X_HUB_SIGNATURE')

    if header_signature is None:
        return HttpResponseForbidden('Permission denied.')

    sha_name, signature = header_signature.split('=')
    if sha_name != 'sha1':
        return HttpResponseForbidden('Invalid signature.')

    mac = hmac.new(secret, msg=request.body, digestmod=hashlib.sha1)

    if not hmac.compare_digest(mac.hexdigest(), signature):
        return HttpResponseForbidden('Invalid signature.')

    # If signatures match, pull the latest changes from GitHub
    subprocess.Popen(['./restart_script.sh'])
    return HttpResponse('Success')