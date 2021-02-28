from ...fastai1.text import TextClassificationInterpretation
from ...fastai1.basics import *
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

class Evaluator():
  def __init__(self):
    pass

  def evaluate_ensemble(self, learn_clas_fwd, learn_clas_bwd):

    preds_fw, y_fw, losses_fw = learn_clas_fwd.get_preds(with_loss=True)
    preds_bw, y_bw, losses_bw = learn_clas_bwd.get_preds(with_loss=True)

    # if abs(accuracy(preds_fw, y_fw) - accuracy(preds_bw, y_bw) > 10):
    #   print("Higher difference between accuracies of fw and bwd models...")

    preds = (preds_fw + preds_bw) / 2
    y = (y_fw + y_bw) / 2
    losses = (losses_fw + losses_bw) / 2

    acc = accuracy(preds, y)
    print('The accuracy is {0} %.'.format(acc))

    err = error_rate(preds, y)
    print('The error rate is {0} %.'.format(err))

    # probs from log preds
    probs = np.exp(preds[:, 1])
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)

    # Compute ROC area
    roc_auc = auc(fpr, tpr)
    print('ROC area is {0}'.format(roc_auc))

    xlim = [-0.01, 1.0]
    ylim = [0.0, 1.01]

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='Ensemble Classifier : ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('False Positive(FP) Rate')
    plt.ylabel('True Positive(TP) Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # use learn_clas_fwd for the ensemble to get close confusion matrix for the actual
    interpretation = ClassificationInterpretation(learn_clas_fwd, preds, y, losses)
    interpretation.plot_confusion_matrix()

    pred_val = learn_clas_fwd.get_preds(DatasetType.Valid, ordered=True)
    pred_val_l = pred_val[0].argmax(1)

    print(classification_report(pred_val[1], pred_val_l))


  def evaluate(self, learn):
    preds, y, losses = learn.get_preds(with_loss=True)

    acc = accuracy(preds, y)
    print('The accuracy is {0} %.'.format(acc))

    err = error_rate(preds, y)
    print('The error rate is {0} %.'.format(err))

    interp = ClassificationInterpretation(learn, preds, y, losses)
    interp.plot_confusion_matrix()

    pred_val = learn.get_preds(DatasetType.Valid, ordered=True)
    pred_val_l = pred_val[0].argmax(1)

    print(classification_report(pred_val[1], pred_val_l))

    # probs from log preds
    probs = np.exp(preds[:, 1])
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)

    # Compute ROC area
    roc_auc = auc(fpr, tpr)
    print('ROC area is {0}'.format(roc_auc))

    xlim = [-0.01, 1.0]
    ylim = [0.0, 1.01]

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    interp2 = TextClassificationInterpretation.from_learner(learn)
    interp2.show_top_losses(10)

    # print(interp2.show_intrinsic_attention(
    #     "ඉඩකඩ සම්බන්ධයෙන් මතු වූ ගැටලුව මහර බන්ධනාගාරයේ කලහකාරී තත්ත්වයට එකහෙළා බලපෑ බව, සිද්ධිය පිළිබඳව විමර්ශනය කිරීම සඳහා අධිකරණ අමාත්‍යවරයා පත්කළ කමිටුවේ අතුරු වාර්තාව පෙන්වා දී තිබේ."))

    torch.argmax(preds[0])