from numpy import savez_compressed;
from numpy import load;
from numpy import expand_dims;
from numpy import asarray;
from matplotlib import pyplot;
from sklearn.metrics import accuracy_score;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import Normalizer;
from sklearn.svm import SVC;
from skimage.transform import resize;
from skimage.io import imread;
from keras.models import load_model;
import joblib;
import makeDataset;
import embedData;
from mtcnn.mtcnn import MTCNN
detector = MTCNN();

facenet = load_model('facenet_keras.h5');
data = load("dataset.npz");
train_images, train_lables, val_images, val_lables = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'];
# normalize the training vectors
in_encoder = Normalizer(norm="l2");
train_images = in_encoder.transform(train_images);
val_images = in_encoder.transform(val_images);

# labe the encoded images

out_encoder = LabelEncoder();
out_encoder.fit(train_lables);
train_lables = out_encoder.transform(train_lables);
val_lables = out_encoder.transform(val_lables);


model = SVC(kernel="linear", probability=True);
model.fit(train_images, train_lables);
joblib.dump(model, "model.sav", compress=0);





