
import numpy as np
import random
from glob import glob
import os
import SimpleITK as sitk
from evaluation_metrics import *
from model import Unet_model


class Prediction(object):


    # constructor
    def __init__(self, batch_size_test, load_model_path):
        self.batch_size_test = batch_size_test
        unet = Unet_model(img_shape=(240, 240, 4), load_model_weights = load_model_path)
        self.model = unet.model
        print("u-net cnn model compiled\n")


    '''
    segment the input volume
    input: (1) str 'filepath_image': filepath of the volume to predict
     (2) bool 'show'
    output: (1) np array of the predicted volume
      (2) np array of the corresponding ground truth
    '''
    def predict_volume(self, filepath_image, show):

        '''
        segment the input volume
        INPUT   (1) str 'filepath_image': filepath of the volume to predict
                (2) bool 'show': True to ,
        OUTPUt  (1) np array of the predicted volume
                (2) np array of the corresping ground truth
        '''

        #read the volume
        flair = glob(filepath_image + '/*_flair.nii.gz')
        t2 = glob(filepath_image + '/*_t2.nii.gz')
        gt = glob(filepath_image + '/*_seg.nii.gz')
        t1s = glob(filepath_image + '/*_t1.nii.gz')
        t1c = glob(filepath_image + '/*_t1ce.nii.gz')
        if (len(flair)+len(t2)+len(gt)+len(t1s)+len(t1c))<5:
            print("there is a problem here! the problem lies in this patient")
            return None, None
        scans_test = [flair[0], t1s[0], t1c[0], t2[0], gt[0]]
        # read a volume composed of 5 modalities
            # print(scans)
        test_im = []
        for k in range(len(scans_test)):
            img = np.array(nib.load(scans_test[k]).dataobj)
            test_im.append(img)


        test_im=np.array(test_im).astype(np.float32)
        test_image = test_im[0:4]
        gt=test_im[-1]
        gt[gt==4]=3
        gt = np.transpose(gt, (2, 1, 0))

        #normalize each slice following the same scheme used for training
        test_image = np.transpose(test_image, (0, 3, 2, 1))
        test_image=self.norm_slices(test_image)
        # print(test_image.shape)

        #transform the data to channels_last keras format
        test_image = test_image.swapaxes(0,1)
        # print(test_image.shape, 'after swapping axes')
        test_image=np.transpose(test_image,(0,2,3,1))
        # print(test_image.shape, 'after transposing')

        if show:
            verbose=1
        else:
            verbose=0
        # predict classes of each pixel based on the model
        prediction = self.model.predict(test_image,batch_size=self.batch_size_test,verbose=verbose)
        prediction = np.argmax(prediction, axis=-1)
        prediction=prediction.astype(np.uint8)
        #reconstruct the initial target values .i.e. 0,1,2,4 for prediction and ground truth
        prediction[prediction==3]=4
        gt[gt==3]=4

        return np.array(prediction),np.array(gt)


    '''
    computes the evaluation metrics on the segmented volume
    input: (1) str 'filepath_image': filepath to test image for segmentation, including file extension
     (2) bool 'save': whether to save to disk or not
     (3) bool 'show': if true, prints the evaluation metrics
    output: np array of all evaluation metrics
    '''
    def evaluate_segmented_volume(self, filepath_image, save, show, save_path):
        predicted_images, gt = self.predict_volume(filepath_image, show)
        if predicted_images is None:
            return None

        if save:
            np.save('/Users/kaitlinylim/Documents/tumorproj/predictions/path_all_train/{}.npy'.format(save_path), predicted_images)
            # tmp = sitk.GetImageFromArray(predicted_images)
            # sitk.WriteImage(tmp,'/Users/kaitlinylim/Documents/tumorproj/predictions/{}.nii.gz'.format(save_path))

        # compute the evaluation metrics
        Dice_complete = DSC_whole(predicted_images, gt)
        Dice_enhancing = DSC_en(predicted_images, gt)
        Dice_core = DSC_core(predicted_images, gt)

        Sensitivity_whole = sensitivity_whole(predicted_images, gt)
        Sensitivity_en = sensitivity_en(predicted_images, gt)
        Sensitivity_core = sensitivity_core(predicted_images, gt)

        Specificity_whole = specificity_whole(predicted_images, gt)
        Specificity_en = specificity_en(predicted_images, gt)
        Specificity_core = specificity_core(predicted_images, gt)

        Hausdorff_whole = hausdorff_whole(predicted_images, gt)
        Hausdorff_en = hausdorff_en(predicted_images, gt)
        Hausdorff_core = hausdorff_core(predicted_images, gt)

        if show:
            print("************************************************************")
            print("Dice complete tumor score : {:0.4f}".format(Dice_complete))
            print("Dice core tumor score (tt sauf vert): {:0.4f}".format(Dice_core))
            print("Dice enhancing tumor score (jaune):{:0.4f} ".format(Dice_enhancing))
            print("**********************************************")
            print("Sensitivity complete tumor score : {:0.4f}".format(Sensitivity_whole))
            print("Sensitivity core tumor score (tt sauf vert): {:0.4f}".format(Sensitivity_core))
            print("Sensitivity enhancing tumor score (jaune):{:0.4f} ".format(Sensitivity_en))
            print("***********************************************")
            print("Specificity complete tumor score : {:0.4f}".format(Specificity_whole))
            print("Specificity core tumor score (tt sauf vert): {:0.4f}".format(Specificity_core))
            print("Specificity enhancing tumor score (jaune):{:0.4f} ".format(Specificity_en))
            print("***********************************************")
            print("Hausdorff complete tumor score : {:0.4f}".format(Hausdorff_whole))
            print("Hausdorff core tumor score (tt sauf vert): {:0.4f}".format(Hausdorff_core))
            print("Hausdorff enhancing tumor score (jaune):{:0.4f} ".format(Hausdorff_en))
            print("***************************************************************\n\n")

        return np.array((Dice_complete, Dice_core, Dice_enhancing,
                    Sensitivity_whole, Sensitivity_core, Sensitivity_en,
                    Specificity_whole, Specificity_core, Specificity_en,
                    Hausdorff_whole, Hausdorff_en, Hausdorff_en))


    '''
    returns an np array of predicted volumes
    '''
    def predict_multiple_volumes(self, filepath_volumes, save, show):
        results, ids = [], []
        for patient in filepath_volumes:
            tmp1 = patient.split('/')
            print("volume id:", tmp1[-2] + '/' + tmp1[-1])
            # might need to change save_path
            tmp=self.evaluate_segmented_volume(patient,save=save,show=show,save_path=os.path.basename(patient))
            if tmp is None:
                continue
            # save the results of each volume
            results.append(tmp)
            # save each id for later use
            ids.append(str(tmp1[-2] + '/' + tmp1[-1]))

        res = np.array(results)
        print("mean:", np.mean(res, axis = 0))
        print("std:", np.std(res, axis = 0))
        print("median:", np.median(res, axis = 0))
        print("25 quartile:", np.percentile(res, 25, axis = 0))
        print("75 quartile:", np.percentile(res, 75, axis = 0))
        print("max:", np.max(res, axis = 0))
        print("min:", np.min(res, axis = 0))

        # might need to change
        np.savetxt("/Users/kaitlinylim/Documents/tumorproj/predictions/path_all_train/results.out", res)
        np.savetxt("/Users/kaitlinylim/Documents/tumorproj/predictions/path_all_train/volumes_id.out", ids, fmt='%s')
        return res


    '''
    normalizes each slice, excluding gt
    subtracts mean and div by std dev for each slice
    clips top and bottom one percent of pixel intensities
    '''
    def norm_slices(self, slice_not):
        normed_slices = np.zeros((4, 155, 240, 240))
        for slice_ix in range(4):
            normed_slices[slice_ix] = slice_not[slice_ix]
            for mode_ix in range(155): #change this back to 155 if it doesn't work
                normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])
        return normed_slices


    def _normalize(self, slice):
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]

        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            tmp[tmp == tmp.min()] = -9
        return tmp


if __name__ == '__main__':


    # set arguments
    model_to_load = '/Users/kaitlinylim/Documents/tumorproj/weights/Res_Unet.epoch_02.hdf5'
    brain_seg_pred = Prediction(batch_size_test = 2, load_model_path = model_to_load)

    path_all_test = [line.rstrip('\n') for line in open('path_all_test.txt')]

    # predict multiple volumes
    final_vol_stats = brain_seg_pred.predict_multiple_volumes(path_all_test, save = True, show = True)
