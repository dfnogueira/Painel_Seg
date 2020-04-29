# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:21:10 2020

@author: dst3834
"""
import gmaps
import requests
import matplotlib.pyplot as plt
import pickle
from PIL import Image#, ImageTk
from keras import backend as K
from segmentation_models import Unet
import image_slicer
import numpy as np
import os.path
import copy
import cv2
import tensorflow as tf
from pathlib import Path
import config

def Proxy(login, passw, API_KEY, zoom,scale,coor):
    
    login = config.login
    passw = config.passw
    API_KEY = config.API_KEY
    
    http  = "http://"  + login + ":" + passw + "@proxy.dst.local:8080"
    https = "https://" + login + ":" + passw + "@proxy.dst.local:8080"
    
    proxies = {
            "http":http,
            "https":https
            }
    gmaps.configure(api_key=API_KEY)

    url = "https://maps.googleapis.com/maps/api/staticmap?center="+coor+"&zoom="+str(zoom)+"&size=640x640&maptype=satellite&scale="+str(scale)+"&key="+API_KEY

    return proxies,url

def Save_image(filename,r):
    # Save image
    f = open(filename, 'wb') 
    # r.content gives content, in this case gives image 
    f.write(r.content) 
    f.close() 

def LoadVar(dir_var, name):
    # Getting back the objects:
    filename2 = dir_var + name
    f2 = open(filename2, 'rb')
    temp = pickle.load(f2)
    f2.close()
    return temp

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def Dividir_Imagem(img, tam):
    
    x = img.shape[0]
        
    div = int(np.round(x/tam,0))
    
    slc = div**2
    
    dir_img = './static/'
    filename = dir_img + 'test.png'
    
    tiles = image_slicer.slice(filename, slc, save=False)    
    
    dir_slice = dir_img + 'div/'
    image_slicer.save_tiles(tiles,
                            directory=dir_slice,
                            prefix='slice', format='png')

    files = os.listdir(dir_slice)
    
    data = list()
    for file in files:
        filename = dir_slice+file
        data.append(plt.imread(filename))    
        
    dd = list()
    for i in range(len(data)):
         dd.append((data[i] * 255).round().astype(np.uint8))    
    
    ddd = np.zeros((len(dd),dd[0].shape[0],dd[0].shape[1],dd[0].shape[2]),dtype=dd[0].dtype)
    for i in range(len(dd)):
        ddd[i,:,:,:] = dd[i]
        
    return ddd, files, dir_slice, div
    
def Contorno(inicio, fim, y_pred, x_test, fator,Plot = True):
        res_imag = np.array([])
        res_pred = np.array([])
        contorno = list()
        img_ctn = list()
        ind_com_contorno = list()
        for i in range(inicio,fim):
            print('Imagem ' + str(i))
            contours_pred,_ = cv2.findContours(cv2.convertScaleAbs(y_pred[i]), 
                                                       cv2.RETR_EXTERNAL, 
                                                       cv2.CHAIN_APPROX_NONE)
    
            img_conj = copy.deepcopy(x_test[i][:,:,0:3])
            
            count_pred = 0
            for c in contours_pred:
                area = cv2.contourArea(c)
                if area > fator:
                    ind_com_contorno.append(i) 
                    (x, y, alt, lar) = cv2.boundingRect(c)
                    cv2.rectangle(img_conj, (x, y), (x+alt, y+lar), (255, 0, 255), 1)
                    contorno.append(c)
                    count_pred = count_pred+1
            
            img_ctn.append(img_conj)
                    
            res_imag  = np.append(res_imag,i)
            res_pred  = np.append(res_pred,count_pred)
            
            if Plot == True:
                print ('quantidade de paineis preditas = ' + str(count_pred))
                plt.figure(figsize=(10,10))
                plt.title('Conj')
                plt.imshow(img_conj[:,:,0:3])
                plt.tight_layout()
                plt.show()
                
        dados = [res_imag,res_pred]
    
        return dados, contorno, img_ctn, np.unique(ind_com_contorno)       
 

def MontarImagem(tam, div,img_ctn,x_test,y_pred):
    image = np.zeros((div*tam,div*tam,3), dtype='uint8')
    label = np.zeros((div*tam,div*tam))
    cnt   = np.zeros((div*tam,div*tam,3), dtype='uint8')

    ini = list()
    fim = list()  
    for i in range(0,div):
        ini.append(i*tam)
        fim.append(i*tam+(tam))
    
    cont = 0    
    for i in range(0,div):
        for j in range(0,div):
            image[ini[i]:fim[i],ini[j]:fim[j],:] = x_test[cont]
            label[ini[i]:fim[i],ini[j]:fim[j]] = y_pred[cont]
            cnt[ini[i]:fim[i],ini[j]:fim[j],:] = img_ctn[cont]
            cont = cont+1
            
    return image, label, cnt

def Contar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, np.median(gray), np.max(gray))
    
    kernel = np.ones((2,2),np.uint8)
    ll_cl = cv2.dilate(canny,kernel,iterations = 1)
   
    ll_cp = 1*(ll_cl>np.mean(ll_cl)) 
    
    cnts = cv2.findContours(cv2.convertScaleAbs(ll_cp),
                            cv2.RETR_LIST, 
                            cv2.CHAIN_APPROX_SIMPLE)
    
    aa = list()
    for c in cnts:
        if len(c) > 1:
            for cc in c:
                area = cv2.contourArea(cc)
                aa.append(area)

    img_conj = copy.deepcopy(image)
    contador = 0
    for c in cnts:
        if len(c) > 1:
            for cc in c:
                area = cv2.contourArea(cc)
                if (area > 180)&(area < 350):
                    contador += 1 
                    (x, y, alt, lar) = cv2.boundingRect(cc)
                    cv2.rectangle(img_conj, (x, y), (x+alt, y+lar), (255, 0, 255), 3)
   
    print(contador)
    
    return contador

def ML(lat,lon):
    my_file = Path("./static/test.png")
    if my_file.is_file():
         os.remove("./static/test.png")
         print('Excluir TEST.PNG')

    my_file = Path("./static/resultado.png")
    if my_file.is_file():
         os.remove("./static/resultado.png")
         print('Excluir RESULTADO.PNG')
    
    global coor     
    coor = lat + ',' + lon
    zoom  = 20
    scale = 4
    filename = './static/test.png'
        
    proxies, url = Proxy(zoom,scale,coor)   
        
    print('Carregando a imagem')
    r = requests.get(url, proxies = proxies)
       
    print('Salvando a imagem')
    Save_image(filename,r)
    
    #######################################################################
    tam = 256
        
    img = plt.imread(filename)  
    
    print('Dividindo a imagem para 256x256')
    x_test, files, dir_slice, div = Dividir_Imagem(img,tam)
        
    # Predict segmentation
    print('Aplicando o modelo')
    
    global graph    
    with graph.as_default():
        yy_pred = model.predict(x_test)
        
        y_pred = list()
        for i in range(len(yy_pred)):
            y_pred.append(yy_pred[i][:,:,0])
        
        cont = 0
        for pred in y_pred:
            filename = './static/pred/pred_%d.png'%cont
            plt.imsave(filename, pred)
            cont += 1            
                
        dado,contorno,img_cnt, ind_com_contorno = Contorno(0, len(y_pred), y_pred, x_test[:,:,:,0:3], 20, Plot = False) 
       
        image_list = list()
        files = os.listdir('./static/div/')
        for file in files:
            filename = './static/div/%s'%file
            image_list.append(cv2.imread(filename))
     
        contador = list()
        for ind in ind_com_contorno:
            image = image_list[ind]
            # Uniformizar o label     
            contador.append(Contar(image))
        
        image, label, img_contorno = MontarImagem(tam, div,img_cnt,x_test,y_pred)
    
        ll = np.round(label,0)
    
        kernel = np.ones((5,5),np.uint8)
        ll_cl = cv2.dilate(ll,kernel,iterations = 10)
                    
        contours_pred,_ = cv2.findContours(cv2.convertScaleAbs(ll_cl), 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_NONE)
                
        img_conj = copy.deepcopy(image)
                    
        fator = 20
            
        for c in contours_pred:
                    area = cv2.contourArea(c)
                    if area > fator:
                        (x, y, alt, lar) = cv2.boundingRect(c)
                        cv2.rectangle(img_conj, (x, y), (x+alt, y+lar), (255, 0, 255), 10)
            
        print('Salvar resultado')
        filename_res = './static/resultado.png'
        im = Image.fromarray(img_conj)
        im.save(filename_res)
            
    return coor, contador
##################################################################################
from flask import Flask, render_template, request
from flask_caching import Cache

# Desabilitar o cache
cache = Cache(config={'CACHE_TYPE': 'null'})

app = Flask(__name__)

app.config['CACHE_TYPE'] = 'null'
cache.init_app(app)

# Load model pretrained
print('Carregando o modelo')
    
model = Unet('resnet34')

graph = tf.get_default_graph() 
        
model_filename = 'weights_model_IOU_DICE.h5'
        
model.load_weights(model_filename)

model.summary()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def imagem_post():
    lat = request.form['lat']
    lon = request.form['lon']
    
    coor, contador = ML(lat,lon)
    url_tes = "./static/test.png?"+coor
    url_res = "./static/resultado.png?"+coor
    
    return render_template('imagem.html', texto=coor, quant=np.sum(contador),url_tes=url_tes,url_res=url_res)

@app.route("/static")
def get_img(coor):
    return 'test.png'

@app.route("/static")
def get_res(coor):
    return 'resultado.png'

if __name__ == "__main__":
    app.run()
