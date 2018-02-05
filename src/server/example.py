import os
import imp
import sys
import time
import gzip
from StringIO import StringIO
from Cheetah.Template import Template
import chardet
import base64
import recognizer
def tt(request,response_head):
    print "get request"
    #print type(request.form['img'].value)
    #print request.getdic
    #print request.postdic
    #f = request.files['file']
   # print len(request.form['img'])
   # dic = request.form['img'].environ
   # print(request.form['img'])
    s = request.form['img'].value
    print len(s)
    f = request.form['img'].fp
    while True:
        t = f.readline()
        if t=='':
            print('end')
            break
        print(t.strip())
    #s.decode('ISO-8859-1')
    file = open('test.jpg','wb')
    file.write(s)
    file.close()
    
    #print type(s)
    #s.encode(encoding='ISO-8859-1').decode('gbk')
    #s.encode(encoding='ISO-8859-1')
    #sb = str.encode(s,encoding='ISO-8859-1')
    #print isinstance(s,unicode)
    #print chardet.detect(s)
    #str.encode(s)
    #str.encode(s,'')
    #print isinstance(s,unicode)
    #imgdata = base64.b64decode(s)
    #file = open('test.jpg','wb')
    #file.write(imgdata)
    #file.close()
    return "ccb"+request.path
def getdata(request,response_head):
    f=open("a.txt")
    content = f.read()
    f.close()
    response_head["Content-Encoding"] = "gzip"
    return content
def template(request,response_head):
    t = Template(file="template.html")
    t.title  = "my title"
    t.contents  = "my contents"
    response_head["Content-Encoding"] = "gzip"
    return str(t)
