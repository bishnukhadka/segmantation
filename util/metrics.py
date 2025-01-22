import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        """
        Returns the mIoU score of all the classes excluding the background.
        Assumes the background is label=0.
        """
        miou_scores = []
        epsilon = 1e-7  # Small value to prevent division by zero
        
        for i in range(self.num_class):  # Include the last class
            tp = self.confusion_matrix[i, i]  # True positives
            fp = np.sum(self.confusion_matrix[:, i]) - tp  # False positives
            fn = np.sum(self.confusion_matrix[i, :]) - tp  # False negatives
            
            denominator = tp + fp + fn + epsilon
            if denominator > epsilon:  # Avoid adding NaN for empty classes
                miou = tp / denominator
                miou_scores.append(miou)
        
        if len(miou_scores) == 0:
            return 0.0  # Return 0 if no valid classes
        return np.nanmean(miou_scores)
        

    def get_dice_score(self):
        """
        returns the dice score of all the classes excluding bacground
        Assuming the background is label=0
        """
        dice_scores = []
        epsilon = 1e-7
        for i in range(self.num_class):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            dice_score = (2 * tp) / ((2 * tp + fp + fn) + epsilon)
            dice_scores.append(dice_score)

        if len(dice_scores) == 0:
            return 0.0

        return np.nanmean(dice_scores)
    
    def get_precision_and_recall(self):
        precisions = []
        recalls = []
        epsilon = 1e-7
        for i in range(self.num_class):
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            precisions.append(precision)
            recalls.append(recall)
        precision, recall = np.mean(precisions), np.mean(recalls)
        if len(precisions) == 0:
            precision=0.0
        if len(recalls) == 0:
            recall= 0.0
        return precision, recall
    
    def Frequency_Weighted_Intersection_over_Union(self):
        epsilon = 1e-7
        # coded for semantic segmentation. 
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix) + epsilon)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    # def _generate_matrix(self, gt_image, pred_image):
    #     """
    #     Generate confusion matrix for one batch.
    #     :param gt_image: Ground truth (true class labels)
    #     :param pred_image: Predicted class labels
    #     :return: Batch confusion matrix
    #     """
    #     # Ensure ground truth and predictions are integers
    #     gt_image = gt_image.astype(np.int64)
    #     pred_image = pred_image.astype(np.int64)

    #     # Create a mask to filter valid class values
    #     mask = (gt_image >= 0) & (gt_image < self.num_class)

    #     # Compute combined labels for bincount
    #     label = self.num_class * gt_image[mask] + pred_image[mask]

    #     # Ensure label is an integer array
    #     label = label.astype(np.int64)

    #     # Compute confusion matrix
    #     count = np.bincount(label, minlength=self.num_class**2)
    #     confusion_matrix = count.reshape(self.num_class, self.num_class)
    #     # print(f'Confusion matrix: {confusion_matrix}')
    #     return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    # def add_batch(self, gt_image, pred_image):
    #     """
    #     Update confusion matrix for a batch of ground truth and predictions.
    #     :param gt_image: Ground truth labels for the batch
    #     :param pred_image: Predicted labels for the batch
    #     """
    #     assert gt_image.shape == pred_image.shape, "Shapes of ground truth and predictions must match."
    #     self.confusion_matrix += self._generate_matrix(gt_image, pred_image)

    def reset(self):
        # self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int64)