# Import libraries
import os
import cv2
import time
import keras
import random
import numpy as np
import matplotlib.pyplot as plt
import Chaka_Faster_rcnn.common as common
import Chaka_Faster_rcnn.Vgg as mb

import  Chaka_Faster_rcnn.layers as layers

# The main entry point for this module
def main():

    # Create configuration
    config = common.Config()
    config.model_path = 'C:/Users/rzouga/Downloads/Github/CNN_CV/Face_Recognition_Fast_rcnn/TrainFacialRecognitonModel/work/work_V2/training_model.h5'
    config.use_horizontal_flips = False
    config.use_vertical_flips = False
    config.rot_90 = False
    config.num_rois = 4

    # Set the gpu to use
    common.setup_gpu('cpu')
    #common.setup_gpu(0)

    # Create a dictionary with classes and switch key values
    classes = {'Noface': 1, 'face': 0}
    mappings = {v: k for k, v in classes.items()}
    class_to_color = {mappings[v]: np.random.randint(0, 255, 3) for v in mappings}
    print(mappings)

    # Get inference models
    model_rpn, model_classifier, model_classifier_only = mb.get_inference_models(config, len(classes), weights_path=config.model_path)

    # Get a list of test images
    images = os.listdir('C:/Users/rzouga/Downloads/Github/CNN_CV/Face_Recognition_Fast_rcnn/TrainFacialRecognitonModel/wiki_crop/wiki_crop/02/')

    # Randomize images to get different images each time
    random.shuffle(images)

    # If the box classification value is less than this, we ignore this box
	#must return to 0.7
    bbox_threshold = 0.3

    # Start visual evaluation
    for i, name in enumerate(images):

        # Limit the evaluation to 4 images
        if i > 4:
            break;

        # Print the name and start a timer
        print(name)
        st = time.time()

        # Get the filepath
        filepath = os.path.join('C:/Users/rzouga/Downloads/Github/CNN_CV/Face_Recognition_Fast_rcnn/TrainFacialRecognitonModel/wiki_crop/wiki_crop/02/', name)

        # Get the image
        img = cv2.imread(filepath)

        # Format the image
        X, ratio = common.format_img(img, config)
        X = np.transpose(X, (0, 2, 3, 1))

        # Get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        [Y1, Y2, F] = model_rpn.predict(X)

        # Get bboxes by applying NMS 
        # R.shape = (300, 4)
        R = layers.rpn_to_roi(Y1, Y2, config, keras.backend.image_data_format(), overlap_thresh=0.7)

        # Convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # Apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0]//config.num_rois + 1):
            ROIs = np.expand_dims(R[config.num_rois*jk:config.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//config.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],config.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):
                # Ignore 'bg' class
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                # Get the class name
                cls_name = mappings[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = layers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([config.rpn_stride*x, config.rpn_stride*y, config.rpn_stride*(x+w), config.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = common.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]

                # Calculate real coordinates on original image
                (real_x1, real_y1, real_x2, real_y2) = common.get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

                textLabel = '{0}: {1}'.format(key,int(100*new_probs[jk]))
                all_dets.append((key,100*new_probs[jk]))

                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1-0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        print('Elapsed time = {0}'.format(time.time() - st))
        print(all_dets)
        plt.figure(figsize=(10,10))
        plt.grid()
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.savefig('C:/Users/rzouga/Downloads/Github/CNN_CV/Face_Recognition_Fast_rcnn/TrainFacialRecognitonModel/work/work_V2/Test_Result/' + str(i) + '.png')
        #plt.show()

# Tell python to run main method
if __name__ == "__main__": main()