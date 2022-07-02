from os import listdir
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
import cv2;
from mtcnn.mtcnn import MTCNN
detector = MTCNN();

#returns an image array closeup of a given face.
def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename);
	imgarr = asarray(image);
	results = detector.detect_faces(imgarr);
	print(filename);
	print(results);

	x1, y1, width, height = results[0]['box'];

	x1, y1 = abs(x1), abs(y1);
	x2, y2 = x1 + width, y1 + height;

	extracted_face = imgarr[y1:y2, x1:x2];
	#imageGS = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2GRAY)
	image = Image.fromarray(extracted_face).resize(required_size);


	#print(f"from extract face: {asarray(image)}")
	return asarray(image);

#uses extract_face to get all the faces from one directory and returns a list of imagearrays.
def load_faces(dir):
	faces = list();

	for filename in listdir(dir):
		path = dir + filename;

		face = extract_face(path);
		faces.append(face);

	return faces;

#loads an entire dataset and returns two array x and y.
#x is an array of all the faces in the directory.
#y is an array of lables.
def load_dataset(dir):
	x, y = list(), list();

	for subdir in listdir(dir):
		path = dir + subdir + '/';
		faces = load_faces(path);
		lables = [subdir for _ in range(len(faces))];

		x.extend(faces);
		y.extend(lables);

	return asarray(x), asarray(y);

def main():
	train_images, train_lables = load_dataset("data/train/");

	val_images, val_lables = load_dataset("data/val/");

	savez_compressed("dataset.npz", train_images, train_lables, val_images, val_lables);

if __name__ == "__main__":
	main();



