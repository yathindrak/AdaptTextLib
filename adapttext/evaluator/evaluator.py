from ...fastai1.text import TextClassificationInterpretation
from ...fastai1.basics import *
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.metrics import roc_curve, auc


class Evaluator():
    """Provide evaluation metrics"""

    def __init__(self):
        pass

    def evaluate(self, learn, learn_cls_fwd):
        """
        Evaluate the ensemble model
        """
        preds, y, losses = learn.get_preds(with_loss=True)

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
        classification_interpretation = ClassificationInterpretation(learn, preds, y, losses)
        classification_interpretation.plot_confusion_matrix()

        pred_val = learn.get_preds(DatasetType.Valid)
        pred_val_l = pred_val[0].argmax(1)

        print(classification_report(pred_val[1], pred_val_l))

        print("--Mathews Correlation Coefficient--")
        print(matthews_corrcoef(pred_val[1], pred_val_l))

        text_classification_interpretation = TextClassificationInterpretation.from_learner(learn_cls_fwd)
        text_classification_interpretation.show_top_losses(10)

    def get_accuracy(self, learn):
        """
        Get model accuracy
        :param learn: model
        :type learn: object
        :return: accuracy
        :rtype: float
        """
        preds, y, losses = learn.get_preds(with_loss=True)

        acc = accuracy(preds, y)
        print('The accuracy is {0} %.'.format(acc))

        return acc
