# Import libraries
import os
import time
import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import Chaka_Faster_rcnn.common as common
import Chaka_Faster_rcnn.Vgg as mb

import  Chaka_Faster_rcnn.layers as layers

# The main entry point for this module
def main():

    # Create configuration
    config = common.Config()
    config.annotations_file_path = 'C:/Users/rzouga/Downloads/Github/CNN_CV/Face_Recognition_Fast_rcnn/TrainFacialRecognitonModel/work/work_V2/Train_annotation.txt'
    config.pretrained_model_path = 'C:/Users/rzouga/Downloads/Github/CNN_CV/Face_Recognition_Fast_rcnn/TrainFacialRecognitonModel/work/work_V2/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    config.model_path = 'C:/Users/rzouga/Downloads/Github/CNN_CV/Face_Recognition_Fast_rcnn/TrainFacialRecognitonModel/work/work_V2/training_model.h5'
    config.records_path = 'C:/Users/rzouga/Downloads/Github/CNN_CV/Face_Recognition_Fast_rcnn/TrainFacialRecognitonModel/work/work_V2/records.csv'
    config.use_horizontal_flips = True
    config.use_vertical_flips = True
    config.rot_90 = True
    config.num_rois = 4

    # Set the gpu to use
    common.setup_gpu('cpu')
    #common.setup_gpu(0)

    # Get data
    st = time.time()
    images, classes, mappings = common.get_data(config)
    print()
    print('Spend {0:0.2f} mins to load the data'.format((time.time()-st)/60))

    # Make sure that we have a bg class. A background region, this is usually for hard negative mining
    if 'bg' not in classes:
        classes['bg'] = 0
        mappings['bg'] = len(mappings)

    # Print class distribution
    print('Training images per class:')
    print(classes)
    print('Number of classes (including bg) = {}'.format(len(classes)))
    print(mappings)

    # Get a train generator
    train_generator = common.get_anchor_gt(images, config, mode='train')

    # Get training models
    if os.path.isfile(config.model_path):
    
        # Get training models
        model_rpn, model_classifier, model_all = mb.get_training_models(config, len(classes), weights_path=config.model_path)

        # Load records
        record_df = pd.read_csv(config.records_path)
        r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
        r_class_acc = record_df['class_acc']
        r_loss_rpn_cls = record_df['loss_rpn_cls']
        r_loss_rpn_regr = record_df['loss_rpn_regr']
        r_loss_class_cls = record_df['loss_class_cls']
        r_loss_class_regr = record_df['loss_class_regr']
        r_curr_loss = record_df['curr_loss']
        r_elapsed_time = record_df['elapsed_time']
        r_mAP = record_df['mAP']

        print('Already trained {0} batches'.format(len(record_df)))

    else:
        # Check if we can load weights from pretrained model or not
        if os.path.isfile(config.pretrained_model_path):
            model_rpn, model_classifier, model_all = mb.get_training_models(config, len(classes), weights_path=config.pretrained_model_path)
        else:
            model_rpn, model_classifier, model_all = mb.get_training_models(config, len(classes))

        # Create a records dataframe
        record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])

    # Settings for training
    total_epochs = len(record_df)
    r_epochs = len(record_df)
    steps = 240 # 240 images (epoch_length)
    num_epochs = 1
    iter_num = 0
    total_epochs += num_epochs
    losses = np.zeros((steps, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    if len(record_df)==0:
        best_loss = np.Inf
    else:
        best_loss = np.min(r_curr_loss)

    # Start training (one image on each iteration)
    start_time = time.time()

    # Loop epochs
    for epoch_num in range(num_epochs):

        # Create a progress bar
        progbar = keras.utils.Progbar(steps)

        # Print the current epoch
        print('Epoch {}/{}'.format(r_epochs + 1, total_epochs))
        r_epochs += 1

        while True:
            try:

                if len(rpn_accuracy_rpn_monitor) == steps and config.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                # Get the next element from an generator
                # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
                X, Y, img_data, debug_img, debug_num_pos = next(train_generator)

                # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
                loss_rpn = model_rpn.train_on_batch(X, Y)

                # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
                P_rpn = model_rpn.predict_on_batch(X)

                # R: bboxes (shape=(300,4))
                # Convert rpn layer to roi bboxes
                R = layers.rpn_to_roi(P_rpn[0], P_rpn[1], config, keras.backend.image_data_format(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
            
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                # X2: bboxes that iou > config.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
                # Y1: one hot code for bboxes from above => x_roi (X)
                # Y2: corresponding labels and corresponding gt bboxes
                X2, Y1, Y2, IouS = common.calc_iou(R, img_data, config, mappings)

                # If X2 is None means there are no matching bboxes
                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue
            
                # Find out the positive anchors and negative anchors
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if config.num_rois > 1:
                    # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                    if len(pos_samples) < config.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, config.num_rois//2, replace=False).tolist()
                
                    # Randomly choose (num_rois - num_pos) neg samples
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=True).tolist()
                
                    # Save all the pos and neg samples in sel_samples
                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                # Train classifier
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]
                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]
                iter_num += 1

                # Update the progress bar
                progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                            ('final_cls', np.mean(losses[:iter_num, 2])), ('final_regr', np.mean(losses[:iter_num, 3]))])
            
                # Free memory before each step
                #tf.keras.backend.clear_session()

                # Start the next step
                if iter_num == steps:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if config.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {0}'.format(mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {0}'.format(class_acc))
                        print('Loss RPN classifier: {0}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {0}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {0}'.format(loss_class_cls))
                        print('Loss Detector regression: {0}'.format(loss_class_regr))
                        print('Total loss: {0}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                        print('Elapsed time: {0}'.format(time.time() - start_time))
                        elapsed_time = (time.time()-start_time)/60

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    # Save the model
                    if curr_loss < best_loss:
                        if config.verbose:
                            print('Total loss decreased from {0} to {1}, saving model.'.format(best_loss,curr_loss))
                        best_loss = curr_loss
                        #model_all.save_weights(config.model_path)
                        model_all.save(config.model_path)

                    # Create a new records row
                    new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3), 
                                'class_acc':round(class_acc, 3), 
                                'loss_rpn_cls':round(loss_rpn_cls, 3), 
                                'loss_rpn_regr':round(loss_rpn_regr, 3), 
                                'loss_class_cls':round(loss_class_cls, 3), 
                                'loss_class_regr':round(loss_class_regr, 3), 
                                'curr_loss':round(curr_loss, 3), 
                                'elapsed_time':round(elapsed_time, 3), 
                                'mAP': 0}

                    # Append a new record row and save the file
                    record_df = record_df.append(new_row, ignore_index=True)
                    record_df.to_csv(config.records_path, index=0)

                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                continue

        # Free memory after each epoch
        #tf.keras.backend.clear_session()

    print('Training complete, exiting.')

# Tell python to run main method
if __name__ == "__main__": main()