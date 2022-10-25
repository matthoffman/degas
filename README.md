# Degas
DGA-generated domain detection using deep learning models

![Edgar Degas, "Four Dancers"](degas_four_dancers.jpg)

# Running
I'm currently using Conda (Anaconda/Miniconda) for development, but you should be able to use Pipenv or virtualenv as 
well using the included requirements.txt. 

conda: 
```
conda env create -f environment.yml
conda activate degas
```
Pipenv:
```
pipenv install -r requirements.txt
pipenv shell
```
Make sure  h5py with version 2.10.0 in your env. 

Virtualenv is similar, but there's really no reason to use virtualenv instead of Pipenv anymore.


## Retraining the model
There is a trained model checked into the `models` directory. If you'd like to train your own, you'll first need to 
download the training data from S3: 
```
python degas/runner download-data
```
Process the data into the simple CSV form that the model builder expects: 
```
python degas/runner process-data data/raw data/processed
```
Those steps only need to be run once, unless you change the training data.

To then retrain the model using the generated dataset, first install tensorflow-gpu using your package manager of choice 
(`conda install tensorflow-gpu` or `pip install tensorflow-gpu`) so that training is GPU-accelerated.

Then, run:
```
python degas/runner train-model data/processed
```

Run `python degas/runner train-model --help` for some available tuning options.
This takes about an hour and a half on an GTX 1070. It only runs about 9 epochs before it short-circuits; you could 
potentially run it for, say, 5 epochs and still get good accuracy with half the training time: 
`python degas/runner train-model --epochs 5 data/processed `


## Making predictions

Since this project uses Tensorflow as the underlying deep learning library, the recommended way to use this for 
inference is to use [Tensorflow Serving](https://www.tensorflow.org/serving/). 

You should be able to serve it using:
```
'docker run -p 8501:8501 \
  --mount type=bind,source=/Users/yourUserName/PycharmProjects/degas/models/degas,target=/models/degas\
  -e MODEL_NAME=degas -t tensorflow/serving:1.12.0'
```
See [Tensorflow Serving docs](https://www.tensorflow.org/serving/docker) for more information about available options.

show model info:
http://localhost:8501/v1/models/degas
model metadata:
http://localhost:8501/v1/models/degas/metadata
make a predict:
http://localhost:8501/v1/models/degas:predict
post json is :
{
  "instances": [ "www.google.com", "www.a2x43v89es0-1.com", "www.twitter.com" ]
}

# About Degas

Why deep learning for this task? Because it works well, and it isn't hard to implement. From [Byu et al, 2018](http://faculty.washington.edu/mdecock/papers/byu2018a.pdf): 

> "Deep neural networks have recently appeared in the literature on DGA detection Woodbridge et al. (2016); Saxe & Berlin (2017); Yu et al. (2017). They significantly outperform traditional machine learning methods in accuracy, at the price of increasing the complexity of training the model and requiring larger datasets."

Since there's plenty of data available to train with, creating a deep learning model is just as easy or easier than the alternatives.


## References
https://openreview.net/forum?id=BJLmN8xRW&noteId=BJLmN8xRW
http://faculty.washington.edu/mdecock/papers/byu2018a.pdf


## Why "Degas"? 
 * Because it's more fun working on a project with a name, rather than "DGA-detector" or something. 
 * Perhaps naming the project after an impressionist painter will make it sound more impressive?
 * It was the first result from the classic "Samba naming algorithm" ( `egrep -i '^d.*g.*a.* /usr/share/dict/words` )

For the record, I'm pronouncing it "de-gah", as in [Edgar Degas](https://en.wikipedia.org/wiki/Edgar_Degas), not "de-gas", as in "to remove all the gas." 
