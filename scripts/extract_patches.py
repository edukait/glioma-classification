
import random
import numpy as np
from glob import glob
import SimpleITK as sitk
from keras.utils import np_utils


class Pipeline(object):


    # constructor
    def __init__(self, list_train, Normalize = True):
        self.scans_train = list_train
        self.train_im = self.read_scans(Normalize)


    '''
    input: unnormalized slice
    output: normalized clipped slice
    '''
    def _normalize(self, slice):
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)
        image_nonzero = slice[np.nonzero(slice)]
        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            # since the range of intensities is between 0 and 5000, the min in the
            # normalized slice corresponds to 0 intensity in unnormalized slice
            # the min is replaced with -9 just to keep track of 0 intensities so that
            # we can discard those intensities afterwards when sampling random patches
            tmp[tmp == tmp.min()] = -9
            return tmp


    '''
    normalizes each slice
    subtracts mean and div by std dev for each slice
    clips top and bottom one percent of pixel intensities
    '''
    def norm_slices(self, slice_not):
        normed_slices = np.zeros((5, 155, 240, 240)).astype(np.float32)
        for slice_ix in range(4):
            normed_slices[slice_ix] = slice_not[slice_ix]
            for mode_ix in range(155):
                normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])
        normed_slices[-1] = slice_not[-1]

        return normed_slices


    def read_scans(self, Normalize):
        print("reading scans...")
        train_images = []
        count = 0
        for i in range(len(self.scans_train)):
            if i % 10 == 0:
                print('iteration [{}]'.format(i))
            folder_name = self.scans_train[i][self.scans_train[i].rfind('/')+1:]
            flair = glob(self.scans_train[i] + '/*_flair.nii.gz')
            t2 = glob(self.scans_train[i] + '/*_t2.nii.gz')
            gt = glob(self.scans_train[i] + '/*_seg.nii.gz')
            t1 = glob(self.scans_train[i] + '/*_t1.nii.gz')
            t1c = glob(self.scans_train[i] + '/*_t1ce.nii.gz')

            if (len(flair) + len(t2) + len(gt) + len(t1) + len(t1c)) < 5:
                # print("there's a problem here. problem lies in this patient:", folder_name)
                count = count + 1
                continue
            scans = [flair[0], t1[0], t1c[0], t2[0], gt[0]]

            # read a volume composed of 5 modalities
            # print(scans)
            tmp = []
            for k in range(len(scans)):
                img = np.array(nib.load(scans[k]).dataobj)
                tmp.append(img)

            tmp = np.transpose(tmp, (0, 3, 2, 1))
            # normalize each slice
            if Normalize == True:
                tmp = self.norm_slices(tmp)

            train_images.append(tmp)
            del tmp
        print("finished reading")
        print(count, "invalid folders/patients")
        return np.array(train_images)


    '''
    input:
    num_patches: the total number of sampled patches
    d: this corresponds to the number of channels (1 modality)
    h: height of the patch
    w: width of the patch

    output:
    patches: np array containing the randomly sampled patches
    labels: np array containing the corresponding target patches
    '''
    def sample_patches_randomly(self, num_patches, d, h, w):
        print("begin sample_patches_randomly function")
        patches = []
        labels = []
        count = 0

        # swap axes to make axis 0 represent the modality and axis 1 represent the slice
        # take the ground truth
        gt_im = np.swapaxes(self.train_im, 0, 1)[4]

        # take the flair image as a mask
        msk = np.swapaxes(self.train_im, 0, 1)[0]
        # save the shape of the ground truth to use it afterwards
        tmp_shp = gt_im.shape

        # reshape the mask and the ground truth to a 1D array
        gt_im = gt_im.reshape(-1).astype(np.uint8)
        msk = msk.reshape(-1).astype(np.float32)

        # maintain list of 1D indices while discarding 0 intensities
        indices = np.squeeze(np.argwhere((msk != -9.0) & (msk != 0.0)))
        del msk

        # shuffle the list of indices of the class
        np.random.shuffle(indices)

        # reshape gt_im
        gt_im = gt_im.reshape(tmp_shp)

        # a loop to sample the patches from the images
        i = 0
        pix = len(indices)
        print("sampling patches")
        while (count < num_patches) and (pix > i):
            # randomly choose an index
            ind = indices[i]
            i += 1
            # reshape ind to 3D index
            ind = np.unravel_index(ind, tmp_shp)
            # get the patient and the slice id (might need to rework)
            patient_id = ind[0]
            slice_idx = ind[1]
            p = ind[2:]
            # construct the patch by defining the coordinates
            p_y = (p[0] - (h)/2, p[0] + (h)/2)
            p_x = (p[1] - (w)/2, p[1] + (w)/2)
            p_x = list(map(int, p_x))
            p_y = list(map(int, p_y))

            # take patches from the modalities
            tmp = self.train_im[patient_id][0:4, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
            # take the corresponding label patch (might need to rework later)
            lbl = gt_im[patient_id, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]

            # keep only patches that have the desired size
            if tmp.shape != (d, h, w):
                continue
            patches.append(tmp)
            labels.append(lbl)
            count += 1
        patches = np.array(patches)
        labels = np.array(labels)
        return patches, labels


'''
concatenate two parts in one dataset
this can be avoided if there is enough RAM
'''
def concatenate():
    Y_labels_2 = np.load("y_dataset_second_part.npy").astype(np.uint8)
    X_patches_2 = np.load("x_dataset_second_part.npy").astype(np.float32)
    Y_labels_1 = np.load("y_dataset_first_part.npy").astype(np.uint8)
    X_patches_1 = np.load("x_dataset_first_part.npy").astype(np.float32)

    # concatenate both parts
    X_patches = np.concatenate((X_patches_1, X_patches_2), axis = 0)
    Y_labels = np.concatenate((Y_labels_1, Y_labels_2), axis = 0)
    del Y_labels_2, X_patches_2, Y_labels_1, X_patches_1

    # shuffle the entire dataset
    shuffle = list(zip(X_patches, Y_labels))
    np.random.seed(138)
    np.random.shuffle(shuffle)
    X_patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
    Y_labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
    del shuffle

    np.save("x_training", X_patches.astype(np.float32))
    np.save("y_training", Y_labels.astype(np.uint8))


if __name__ == '__main__':

    # paths to the BraTS 2018 dataset
    path_HGG = glob('Brats2018/Brats18TrainingData/HGG/**')
    path_LGG = glob('Brats2018/Brats18TrainingData/LGG/**')

    # shuffle the datasets
    np.random.seed(2022)
    np.random.shuffle(path_HGG)
    np.random.shuffle(path_LGG)

    # divide the data into 80% training and 20% testing
    train_val_HGG = (int)(len(path_HGG) * 0.8)
    path_HGG_train = path_HGG[0:train_val_HGG]
    path_HGG_test = path_HGG[train_val_HGG:]

    train_val_LGG = (int)(len(path_LGG) * 0.8)
    path_LGG_train = path_LGG[0:train_val_LGG]
    path_LGG_test = path_LGG[train_val_LGG:]

    path_all_train = path_HGG_train + path_LGG_train
    path_all_test = path_HGG_test + path_LGG_test

    # save the path_all_test to disk
    with open('path_all_test.txt', 'w') as f:
        for path in path_all_test:
            f.write('%s\n' % path)

    print("total number of training images is", len(path_all_train))

    # create the pipeline that extracts patches from the MRI images
    pipeline = Pipeline(list_train = path_all_train)

    np.random.seed(1555)
    start = 0
    end = 57
    # set the total number of patches
    # this formula extracts approximately 3 patches per slice
    num_patches = 146 * (end - start) * 3
    # define the size of a patch
    h = 128
    w = 128
    d = 4

    X_patches, Y_labels = pipeline.sample_patches_randomly(num_patches, d, h, w)

    # transform the data into channels_last keras format
    X_patches = np.transpose(X_patches, (0, 2, 3, 1)).astype(np.float32)

    # we do this transformation so that we have four classes when we one-hot encode the targets
    Y_labels[Y_labels == 4] = 3

    # transform to one-hot encoding for keras
    shp = Y_labels.shape[0]
    Y_labels = Y_labels.reshape(-1)
    Y_labels = np_utils.to_categorical(Y_labels).astype(np.uint8)
    Y_labels = Y_labels.reshape(shp, h, w, 4)

    # shuffle the entire dataset
    shuffle = list(zip(X_patches, Y_labels))
    np.random.seed(180)
    np.random.shuffle(shuffle)
    X_patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
    Y_labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
    del shuffle

    print("size of the patches:", X_patches.shape)
    print("size of their corresponding targets:", Y_labels.shape)

    # save the disk to your files
    #np.save( "x_dataset_first_part",Patches )
    #np.save( "y_dataset_first_part",Y_labels)
