from django.shortcuts import render , HttpResponse

# Create your views here.
def index(request):
    # return HttpResponse("TEST SUB")
    return render(request,'index.html')

def about(request):
    return HttpResponse("TEST ABOUT SUB")

def services(request):
    return HttpResponse("TEST SERVICES SUB")

def contact(request):
    return HttpResponse("CONTACT")