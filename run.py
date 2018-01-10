import expr_classifier.run
from expr_classifier.run import train as classifier_train
import metric_vae.run
from metric_vae.run import train as metric_vae_train
import os

if __name__ == '__main__':
    save_dir = 'experiments/'
    for i in range(100):
        if not os.path.exists(save_dir + str(i)):
            save_dir = save_dir + str(i) + '/'
            os.makedirs(save_dir)
            break

    # classifier_train(save_dir)
    metric_vae_train(save_dir)