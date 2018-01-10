# MetricVAE
Training a metric with distance regularization

# Instructions

Download the Toronto Face Dataset (TFD) files from [here](https://www.dropbox.com/sh/rlcc6araq63fxnr/AACAQBEvGmfXKclP1ZMoe3kza?dl=0) and place in a folder named 'data'.

The code in expr_classifer trains a classifier on TFD by expression labels and then outputs an array containing the image of the original data in the penultimate layer of the classifier. This transformation of the data is used to define a metric on TFD which can then by utilized by the code in metric_vae.

The code in metric_vae trains a VAE on TFD along with a modified loss function which includes a distance regularization term which encourages preservation of the distances given by the metric outputted by expr_classifier. Any metric can be used here.

To run the two parts above in sequence use the command
```
python run.py
```


