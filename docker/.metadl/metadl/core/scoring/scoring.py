""" Runs the scoring procedure for the challenge.
It assumes that there exists a ./model_dir folder containing both the 
submission code and the saved learner. 
It will create a folder named ./scoring_output (default) in which a txt file 
will contain the average score over 600 episodes. You can change the folder 
name via the score_dir flag.

Usage example executed from the metadl/ repository : 

python -m metadl.core.scoring.scoring --meta_test_dir=<path_dataset.meta_test>                     
""" 
import os 
from sys import path

import gin
import numpy as np 
from absl import app
from absl import flags 
from absl import logging
import tensorflow as tf
import scipy 

#from metadl import proto
#import metadl.proto as proto
from metadl.data.dataset import DataGenerator
from metadl.core.ingestion.ingestion import get_gin_path, show_dir

FLAGS = flags.FLAGS

flags.DEFINE_string('meta_test_dir', 
                    '/Users/adrian/GitInria/meta-dataset/records/',
                    ('Directory of the meta-test dataset. This directory '
                        + 'should contain records and a json spec file.'))

flags.DEFINE_string('saved_model_dir',
                    './model_dir',
                    ('Directory path that contains the participant\'s code '
                        + 'along with the serialized learner from meta-fit.'))

flags.DEFINE_string('score_dir',
                    './scoring_output',
                    'Path to the score directory.')

tf.random.set_seed(1234)
def NwayKshot_accuracy(predictions, ground_truth, metric):
    """ N-way, K-shot accuracy which corresponds to the accuracy in a
    multi-classification context with N classes.

    Args:
        predictions : tensors, sparse tensors corresponding to the predicted 
            labels.
        ground_truth : tensors, sparse tensors corresponding the ground truth 
            labels.
        metric : keras.metrics , the metric we use to evaluate the 
            classification performance of the meta-learning algorithm. We use 
            the SparseCategoricalAccuracy in this challenge.

    Retruns:
        score : Float, the resulting performance using the given metric.
    """
    ground_truth = tf.expand_dims(ground_truth, axis = 1)
    predictions = tf.expand_dims(predictions, axis = 1)
    logging.debug('Predictions shape : {} - Ground truth shape : {}'.format(
        predictions.shape, ground_truth.shape))

    metric.update_state(ground_truth, predictions)
    score = metric.result()
    logging.debug('An episode score: {}'.format(score))
    metric.reset_states()
    return score
    
def is_one_hot_vector(x, axis=None, keepdims=False):
  """Check if a vector 'x' is one-hot (i.e. one entry is 1 and others 0)."""
  norm_1 = np.linalg.norm(x, ord=1, axis=axis, keepdims=keepdims)
  norm_inf = np.linalg.norm(x, ord=np.inf, axis=axis, keepdims=keepdims)
  return np.logical_and(norm_1 == 1, norm_inf == 1)

def write_score(score, file_score, duration=-1):
    """Write score of the k-th task in the given file_score."""
    file_score.write('set1_score: {:.6f}\n'.format(float(score)))
    file_score.write('Duration: {:.6f}\n'.format(float(duration)))
    
def write_variance(varr, file_score):
    """Write variance of accuracies of the meta-test episodes task in the given 
    file_score."""
    file_score.write('variance: {:.6f}\n'.format(float(varr)))

def extract_elapsed_time(saved_model_dir):
    """ Extracts elapsed time from the metadata file. It corresponds to the 
    meta-training time, the duration of the ingestion process.
    """
    if not os.path.isdir(saved_model_dir): 
        raise ValueError('Saved model directory does not exists.')

    if os.path.isfile(os.path.join(saved_model_dir, 'metadata')):
        with open(os.path.join(saved_model_dir, 'metadata'), 'r') as f : 
            lines = f.readlines()
            for line in lines : 
                splitted_line = line.split(' ')
                for k, word in enumerate(splitted_line): 
                    if 'elapsed' in splitted_line[k]:
                        elapsed_time = float(splitted_line[k+1])
                        return elapsed_time
        
    return -1

def process_task(task):
    """We are using the meta-dataset code to generate episodes from a dataset. 
    Generated episodes have a specific format. Each is processed such that the 
    the support and query sets are ready to be used by the participants. Each
    set is returned as a tf.data.Dataset object.
    The que_labs are kept hidden.

    Returns : 
        support_dataset : tf.data.Dataset containing the support examples and 
            labels.
        query_dataset : tf.data.Dataset containing the query examples
        que_labs : tuple (query_batch_size, 1), the query examples labels 
            i.e. the ground truth labels.
    """
    sup_set = tf.data.Dataset.from_tensor_slices(\
        (task[0][1], task[0][0]))
    dim = task[0][4].shape[1]
    arr = np.arange(dim)
    np.random.shuffle(arr) # shuffling arr
    query_labs = task[0][4]
    query_imgs = task[0][3]
    
    query_labs_s = tf.gather(query_labs, arr, axis=1)
    query_imgs_s = tf.gather(query_imgs, arr, axis=1)

    que_set = tf.data.Dataset.from_tensor_slices(
            (query_labs_s, query_imgs_s)
    )
    new_ds = tf.data.Dataset.zip((sup_set, que_set))
    for ((supp_labs, supp_img), (que_labs, que_img)) \
            in new_ds :

        logging.debug('Supp labs : {}'.format(supp_labs))
        logging.debug('Query labs : {}'.format(que_labs))

        support_set = tf.data.Dataset.from_tensor_slices(\
            (supp_img, supp_labs))
        query_set = tf.data.Dataset.from_tensor_slices(\
            (que_img,))
        support_set = support_set.batch(5)
        query_set = query_set.batch(95)

    return support_set, query_set, que_labs

def scoring(argv):
    """ 
    For each task, load and fit the Learner with the support set and evaluate
    the submission performance with the query set. 
    A directory 'scoring_output' is created and contains a txt file that 
    contains the submission score and duration. Note that the former is the 
    time elapsed during the ingestion program and hence the meta-fit() 
    duration.

    The metric considered here is the Sparse Categorical Accuracy for a 
    5 classes image classification problem.
    """
    del argv
    saved_model_dir = FLAGS.saved_model_dir
    meta_dir = FLAGS.meta_test_dir
    meta_test_dir = os.path.join(meta_dir, 'meta_test')
    logging.info('Meta dir : {}'.format(meta_dir))
    logging.info('Meta test dir : {}'.format(meta_test_dir))
    # Use CodaLab's path `run/input/ref` in parallel with `run/input/res`
    if not os.path.isdir(meta_test_dir): 
        #meta_test_dir = os.path.join(saved_model_dir, os.pardir, 'ref')
        meta_test_dir = os.path.join(saved_model_dir, 'metadataset/meta_test')
    logging.info('New Meta test dir : {}'.format(meta_test_dir))
    code_dir = os.path.join(saved_model_dir, 'code_dir')
    score_dir = FLAGS.score_dir
    cwd_dir = os.getcwd()

    logging.debug(' #### In RUN: #### ')
    for file in os.listdir(cwd_dir):
        logging.debug('File : {} '.format(os.path.join(cwd_dir, file)))

    output_dir = os.path.join(cwd_dir, 'output')
    logging.debug(' #### In Output #### ')
    for file in os.listdir(output_dir):
        logging.debug('File : {}'.format(os.path.join(output_dir, file)))

    private_output_dir = os.path.join(cwd_dir, 'private_output')
    logging.debug(' #### In Private Output #### ')
    for file in os.listdir(private_output_dir):
        logging.debug('File : {}'.format(os.path.join(private_output_dir, file)))

    logging.debug(' #### Input #### :')
    dir_input = os.path.join(saved_model_dir, os.pardir)
    for file in os.listdir(dir_input):
        logging.debug('File : {}'.format(os.path.join(dir_input, file)))

    logging.debug(' #### REF #### :')    
    dir_ref = os.path.join(dir_input, 'ref')
    for file in os.listdir(dir_ref):
        logging.debug('File : {}'.format(os.path.join(dir_ref, file)))

    logging.debug(' #### SCORES #### :')    
    dir_scores = os.path.join(dir_input, 'scores')
    for file in os.listdir(dir_scores):
        logging.debug('File : {}'.format(os.path.join(dir_scores, file)))

    logging.debug(' #### Saved model dir ####')
    for file in os.listdir(saved_model_dir):
        logging.debug('File : {} '.format(os.path.join(saved_model_dir, file)))
    

    path.append(code_dir)
    logging.info('Code_dir: {}'.format(code_dir))

    logging.info('Current directory : {}'.format(os.getcwd()))
    logging.info('Parent directory : {}'.format(os.path.abspath(os.path.join(code_dir, os.pardir))))
    logging.info('Path current running file : {}'.format(__file__))
    for file in os.listdir(code_dir):
        logging.info('File : {} '.format(file))

    from model import MyLearner
    if(os.path.exists(os.path.join(code_dir, 'model.gin'))):
        gin.parse_config_file(os.path.join(code_dir, 'model.gin'))

    logging.info('Ingestion done! Starting scoring process ... ')

    logging.info('Creating the meta-test episode generator ... \n ')
    generator = DataGenerator(path_to_records=meta_test_dir,
                            batch_config=None,
                            episode_config=[128, 5, 1, 19],
                            pool= 'test',
                            mode='episode')
    
    # The fact that we are not specifying episode_config should only be available
    # when we are not in a competition, typically the ss-metadl format. Otherwise,
    # participants can have control over the test protocol. However didn't use it
    # because participants need to specify a valid episode config. -> Don't want
    # to do that
    logging.info('Test generator config : {}'.format(generator.episode_config))
    meta_test_dataset = generator.meta_test_pipeline
    logging.info('Evaluating performance on episodes ... ')

    meta_test_dataset = meta_test_dataset.batch(1)
    meta_test_dataset = meta_test_dataset.prefetch(5)
    learner = MyLearner()
    
    if (not os.path.isdir(score_dir)):
        os.mkdir(score_dir)
    score_file = os.path.join(score_dir, 'scores.txt')
    results = []
    metric = tf.metrics.SparseCategoricalAccuracy()
    nbr_episodes = 600

    for k , task in enumerate(meta_test_dataset) :
        support_set, query_set, ground_truth = process_task(task)
        learner.load(saved_model_dir)
        predictor = learner.fit(support_set)
        predictions = predictor.predict(query_set)
        score = NwayKshot_accuracy(predictions, ground_truth, metric)
        results.append(score)

        #logging.debug('Score on {} : {}'.format(k, score))
        #logging.debug('Results : {}'.format(results[:20]))
        if(k > nbr_episodes):
            break

    with open(score_file, 'w') as f :
        write_score(sum(results)/len(results),
                    f, 
                    extract_elapsed_time(saved_model_dir))
        write_variance(scipy.std(results), f)

    logging.info(('Scoring done! The average score over {} '
        + 'episodes is : {:.3%}').format(nbr_episodes,
                                        sum(results)/len(results))
    )

if __name__ == '__main__':
    np.random.seed(seed=1234)
    tf.get_logger().setLevel('ERROR')
    app.run(scoring)


    