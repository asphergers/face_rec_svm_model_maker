# Important Notes <br />
I learned to do this from an [article](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) on machinelearningmastery.com. They're an amazing resource for everything machine learning and most of what I know came from them.  <br />
<br />
These scripts were made and tested using python 3.7 so success may vary when using other versions of python.
<br /> <br />
I don't plan on updating this repo at all unless I find any major bugs with the scripts while using them.

# Instructions for use <br />
## Collecting faces <br />
In the data folder there are two sub folders, train and val. This is where your training and validation data goes. Make sure the pictures in here are of good qaulity. The model can be pretty accurate even when working with small amounts of data so you should be prioritizing quality on quantity. <br />

## Making the model <br />
In order to make the model you need to create a numpy dataset by running the makeDataset.py file. This will use google's face_net keras model to extract the faces from your training data and compile everything into a single file.<br />
<br />
After the dataset's been compiled you'll need to match all your training data to labels using the embedData.py script. <br />
<br />
Once you've embded the dataset all you need to do is run the main.py script and this will generate a svm model using sklearn's svm library.
