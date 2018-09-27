from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.externals import joblib


lr = joblib.load("model/lr_model.pkl")
pipeline = PMMLPipeline([("lr-classifier", lr)])
sklearn2pmml(pipeline, "lr-classifier.pmml", with_repr=True)
