"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.metric as metric
import pymia.evaluation.writer as writer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, label_set: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True,
                          'label_set': label_set}

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()
    
    # Filter training data based on label_set
    if label_set == 'small_labels':
        small_labels = [0, 3, 4, 5]
        mask = np.isin(labels_train, small_labels)
        data_train = data_train[mask]
        labels_train = labels_train[mask]
        labels_train[np.isin(labels_train, [1, 2])] = 0  # Reassign large labels to background
    elif label_set == 'large_labels':
        large_labels = [0, 1, 2]
        mask = np.isin(labels_train, large_labels)
        data_train = data_train[mask]
        labels_train = labels_train[mask]
        labels_train[np.isin(labels_train, [3, 4, 5])] = 0  # Reassign small labels to background

    warnings.warn('Random forest parameters not properly set.')

    # Define parameter grid
    param_grid = {'n_estimators': [40, 50, 60, 70, 80, 100],
                  'max_depth': [40, 50, 60, 70, 80, 100]}
    # param_grid = {'n_estimators': [80],
    #               'max_depth': [50]}
    forest = sk_ensemble.RandomForestClassifier(class_weight='balanced')
    
    # Create custom scorer for GridSearchCV
    if label_set == 'all_labels':
        labels = [0, 1, 2, 3, 4, 5]
    if label_set == 'small_labels':
        labels = [0, 3, 4, 5]
    if label_set == 'large_labels':
        labels = [0, 1, 2]
        

    scorer = make_scorer(putil.multiclass_dice_coefficient, labels=labels)

    # Initialize grid search with the custom Dice score evaluator function
    grid_search = GridSearchCV(forest, param_grid, scoring=scorer, verbose=2)
    grid_search.fit(data_train, labels_train)
    best_forest = grid_search.best_estimator_
    print('parameters', grid_search.best_params_)

    start_time = timeit.default_timer()
    forest = best_forest
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator(label_set=label_set)

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        predictions = forest.predict(img.feature_matrix[0])
        # print(np.shape(predictions))
        probabilities = forest.predict_proba(img.feature_matrix[0])
        labels = img.feature_matrix[1]
        # print(np.shape(labels))
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)
        # own_dices = putil.multiclass_dice_coefficient(predictions, labels)
        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    post_process_params = {'simple_post': True}
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=True)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')
        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)
        sitk.WriteImage(img.images[structure.BrainImageTypes.T1w], os.path.join(result_dir, images_test[i].id_ + '_T1W.mha'))
        sitk.WriteImage(img.images[structure.BrainImageTypes.T2w], os.path.join(result_dir, images_test[i].id_ + '_T2W.mha'))
        sitk.WriteImage(img.images[structure.BrainImageTypes.GroundTruth], os.path.join(result_dir, images_test[i].id_ + '_GT.mha'))

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exist
    label_suffix = '_small_labels' if label_set == 'small_labels' else '_large_labels' if label_set == 'large_labels' else '_all_labels'
    result_file = os.path.join(result_dir, f'results{label_suffix}_{t}.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    parser.add_argument(
        '--label_set',
        type=str,
        choices=['all_labels', 'small_labels', 'large_labels'],
        default='all_labels',
        help='Specify which label set to use: all_labels (0-5), small_labels (0, 3, 4, 5), or large_labels (0, 1, 2).'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, args.label_set)
