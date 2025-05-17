import numpy as np

class IouHelper:
  iou_sum = 0
  iou_count = 0

  def add_masks(self, mask1, mask2, name):
    intersection_matrix = np.logical_and(mask1, mask2)
    intersection_count = np.sum(intersection_matrix)

    union_matrix = np.logical_or(mask1, mask2)
    union_count = np.sum(union_matrix)

    if union_count > 0:
      self.iou_sum += (intersection_count/union_count)
      self.iou_count += 1
    # else:
    #   print(f"{name}: union == 0")

  def calculate_iou(self):
    if(self.iou_count == 0):
      return -1
    iou_mean = self.iou_sum / self.iou_count
    return iou_mean
  

class DiceHelper:
  dice_sum = 0
  dice_count = 0

  def add_masks(self, mask1, mask2, name):
    intersection_matrix = np.logical_and(mask1, mask2)
    intersection_count = np.sum(intersection_matrix)

    areas_sum = np.sum(mask1) + np.sum(mask2)

    if areas_sum > 0:
      self.dice_sum += (2*intersection_count/areas_sum)
      self.dice_count += 1
    # else:
    #   print(f"{name}: areas_sum == 0")

  def calculate_dice(self):
    if(self.dice_count == 0):
      return -1
    dice_mean = self.dice_sum / self.dice_count
    return dice_mean
  

class RecallHelper:
  recall_sum = 0
  recall_count = 0

  def add_masks(self, mask_pred, mask_truth):
    tp_matrix = np.logical_and(mask_pred, mask_truth)
    tp = np.sum(tp_matrix)

    fn_matrix = np.logical_and(mask_truth, np.logical_not(mask_pred))
    fn = np.sum(fn_matrix)

    denominator = tp + fn
    if denominator == 0:
      # print("não sei oq fazer")
      return

    # else:
    recall = tp / denominator
    self.recall_sum += recall
    self.recall_count += 1

  def calculate_recall(self):
    if(self.recall_count == 0):
      return -1
    return self.recall_sum / self.recall_count


class PrecisionHelper:
  precision_sum = 0
  precision_count = 0

  def add_masks(self, mask_pred, mask_truth):
    tp_matrix = np.logical_and(mask_pred, mask_truth)
    tp = np.sum(tp_matrix)

    fp_matrix = np.logical_and(mask_pred, np.logical_not(mask_truth))
    fp = np.sum(fp_matrix)

    denominator = tp + fp
    if denominator == 0:
      # print("não sei oq fazer")
      return

    # else:
    precision = tp / denominator
    self.precision_sum += precision
    self.precision_count += 1

  def calculate_precision(self):
    if(self.precision_count == 0):
      return -1
    return self.precision_sum / self.precision_count