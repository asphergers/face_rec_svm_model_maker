from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype("float32");
	mean, std = face_pixels.mean(), face_pixels.std();
	face_pixels = (face_pixels - mean) / std;

	samples = expand_dims(face_pixels, axis=0);

	yhat = model.predict(samples);
	print(f"from embed: {yhat[0]}");
	return yhat[0];


def main():
	data = load("dataset.npz");

	training_images, training_lables, val_images, val_lables = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'];

	model = load_model('facenet_keras.h5')

	new_training_images = list();

	for image in training_images:
		embedding = get_embedding(model, image);
		new_training_images.append(embedding);


	new_val_images = list();
	for image in val_images:
		embedding = get_embedding(model, image);
		new_val_images.append(embedding);

	savez_compressed("dataset.npz", new_training_images, training_lables, new_val_images, val_lables);

if __name__ == "__main__":
	main();