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
import pymia.evaluation.writer as writer


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


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, no_post_process: bool, label_set: str):
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

    # initialize evaluator
    evaluator = putil.init_evaluator(label_set=label_set)

    warnings.warn('Random forest parameters not properly set.')
    forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                n_estimators=60,
                                                max_depth=50)

    start_time = timeit.default_timer()
    forest.fit(data_train, labels_train)
    print('Time elapsed:', timeit.default_timer() - start_time, 's')

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

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
        print(np.shape(predictions))
        probabilities = forest.predict_proba(img.feature_matrix[0])
        labels = img.feature_matrix[1].squeeze()
        print(np.shape(labels))
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')
        
        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.float64),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)
        own_dices = putil.multiclass_dice_coefficient(predictions, labels)
        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # ORIGINAL CODE
    # # post-process segmentation and evaluate with post-processing
    # post_process_params = {'simple_post': True}
    # images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
    #                                                  post_process_params, multi_process=True)

    # for i, img in enumerate(images_test):
    #     evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
    #                        img.id_ + '-PP')

    #     # save results
    #     sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
    #     sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)
    #     sitk.WriteImage(img.images[structure.BrainImageTypes.T1w], os.path.join(result_dir, images_test[i].id_ + '_T1W.mha'))
    #     sitk.WriteImage(img.images[structure.BrainImageTypes.T2w], os.path.join(result_dir, images_test[i].id_ + '_T2W.mha'))
    #     sitk.WriteImage(img.images[structure.BrainImageTypes.GroundTruth], os.path.join(result_dir, images_test[i].id_ + '_GT.mha'))
    
    # ADJUSTED TO PARSER
    # Post-process segmentation and evaluate with post-processing, conditionally
    if not no_post_process:
        post_process_params = {'simple_post': True}
        images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                         post_process_params, multi_process=True)

        for i, img in enumerate(images_test):
            evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                               img.id_ + '-PP')

            # Save post-processed results
            sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)

    # Save the original predictions regardless of post-processing
    for i, img in enumerate(images_test):
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        sitk.WriteImage(img.images[structure.BrainImageTypes.T1w], os.path.join(result_dir, images_test[i].id_ + '_T1W.mha'))
        sitk.WriteImage(img.images[structure.BrainImageTypes.T2w], os.path.join(result_dir, images_test[i].id_ + '_T2W.mha'))
        sitk.WriteImage(img.images[structure.BrainImageTypes.GroundTruth], os.path.join(result_dir, images_test[i].id_ + '_GT.mha'))
        
    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    result_file = os.path.join(result_dir, 'results.csv')
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

    # EXTRA PARSERS
    parser.add_argument(
    '--no_post_process',
    action='store_true',
    help='Run the pipeline without post-processing of the segmentation.'
    )

    parser.add_argument(
        '--label_set',
        type=str,
        choices=['all_labels', 'small_labels', 'large_labels'],
        default='all_labels',
        help='Specify which label set to use: all_labels (0-5), small_labels (0, 3, 4, 5), or large_labels (0, 1, 2).'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, args.no_post_process, args.label_set)
