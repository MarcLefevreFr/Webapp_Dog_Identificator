import os
import numpy as np
#import tensorflow as tf
import pickle
import urllib.request
#import tensorflow

from PIL import Image
from flask import Flask, request, render_template, jsonify, url_for, flash, redirect
from werkzeug.utils import secure_filename

from imgaug import augmenters as iaa

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as prepro_in
from tensorflow.keras.applications.efficientnet import preprocess_input as prepro_eff


UPLOAD_FOLDER = 'images/'
ASSETS = "data/"

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'kkjlqksncjhkqshkqfqf'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 

@app.route('/', methods=['POST'])
def upload_image():

	dico = {}

	if len(os.listdir("static/" + UPLOAD_FOLDER)) > 10 :

		for file in os.listdir("static/" + UPLOAD_FOLDER):

			os.remove("static/" + UPLOAD_FOLDER + file)

	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)

	file = request.files['file']

	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)

	if file and allowed_file(file.filename):

		img = Image.open(file)
		w, h = img.size
		ratio = h/w

		new_w = 300
		new_h = int(new_w * ratio)

		new_img = img.resize((new_w, new_h))
		filename = secure_filename(file.filename)
		img_path = os.path.join("static/" + app.config['UPLOAD_FOLDER'], filename)
		#img.save(os.path.join("static/" + app.config['UPLOAD_FOLDER'], filename))
		img.save(img_path)
		new_img.save(os.path.join("static/" + app.config['UPLOAD_FOLDER'], "res_" + filename))
		#print('upload_image filename: ' + filename)
		#flash('Image successfully uploaded and displayed below')

		dico = predicator.identify_dog(img_path)


		#dico["incep"] = [f"Largeur image : {img.size[0]}", f"Hauteur image : {img.size[1]}"]
		#dico["eff"] = [f"Largeur image resizée : {new_img.size[0]}", f"Hauteur image resizée : {new_img.size[1]}"]

		return preds(filename, dico)

	else :

		flash('Allowed image types are - png, jpg, jpeg, gif')
		return redirect(request.url)
 
# @app.route('/display/<filename>')
# def display_image(filename):
#     print('display_image filename: ' + filename)
#     return redirect(url_for("static", filename = UPLOAD_FOLDER + "res_" + filename), code=301)

#@app.route('/preds/<dico>')
def preds(filename, dico):

	path = app.config['UPLOAD_FOLDER'] + "res_" + filename
	return render_template('preds.html', filename = path, dico = dico)



class Dog_Predictor() :

	def __init__(self):

		# self.mod_incep = load_model("static/" + ASSETS + "mod_inception")
		self.mod_eff = load_model("static/" + ASSETS + "mod_eff30_ftb.h5")
		self.dico_labels = pickle.load(open("static/" + ASSETS + "dict_labels.pickle", "rb"))
		self.crop = iaa.size.CropToSquare(position = "center")

	def identify_dog(self, im_path):

		with Image.open(im_path) as image :

			image = np.asarray(image)
			image = self.crop.augment_image(image)
			image = Image.fromarray(image)

			img_in = image.resize((299,299))
			img_in = np.asarray(img_in)
			img_in = prepro_in(img_in)

			img_eff = image.resize((224,224))
			img_eff = np.asarray(img_eff)
			img_eff = prepro_eff(img_eff)

		img_in = np.expand_dims(img_in, 0)
		img_eff = np.expand_dims(img_eff, 0)

		# pred_in = self.mod_incep.predict([img_in])
		pred_eff = self.mod_eff.predict([img_eff])

		# pred_in_l = list(np.squeeze(pred_in))
		# pred_in_sorted = pred_in_l.copy()
		# pred_in_sorted.sort(reverse = True)

		pred_eff_l = list(np.squeeze(pred_eff))
		pred_eff_sorted = pred_eff_l.copy()
		pred_eff_sorted.sort(reverse = True)

		dico_res = {}
		dico_res["incep"] = []
		dico_res["eff"] = []

		# for i in pred_in_sorted[:3]:


		# 	if i > 0.01 :

		# 		dico_res["incep"].append((list(self.dico_labels.keys())[pred_in_l.index(i)], f"{i*100:.2f} % de confiance"))

		for i in pred_eff_sorted[:3]:

			if i > 0.01 :

				dico_res["eff"].append((list(self.dico_labels.keys())[pred_eff_l.index(i)], f"{i*100:.2f} % de confiance"))

		return dico_res
    
# instantiation d'un "prédicteur"...
predicator = Dog_Predictor()




if __name__ == '__main__':

    app.run(debug = True)


