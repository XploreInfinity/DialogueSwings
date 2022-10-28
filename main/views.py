from django.shortcuts import render
from .utils.NLProcessor import NLProcessor
# Create your views here.
#The home page
def upload(request):
    context = {}
    if request.method == 'POST':
        #check if the file has been uploaded:
        if request.FILES:
            chatFile = request.FILES['chatFile']
            nlproc = NLProcessor(chatFile)
            graph,labelledEmojiCount,orderedSenders = nlproc.get_analysis()
            context['messageClassification'] = graph
            context['labelledEmojiCount'] = labelledEmojiCount
            context['orderedSenders'] = orderedSenders
        return render(request,'main/results.html',context=context)
    else:
        return render(request,'main/upload.html')

def home(request):
    return render(request,"main/home.html")
#The about page
def about(request):
    return render(request,"main/about.html")