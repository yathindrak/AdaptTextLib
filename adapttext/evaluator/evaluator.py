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
        ensemble_predictions, y_values, losses = learn.get_preds(with_loss=True)

        ensemble_accuracy = accuracy(ensemble_predictions, y_values)
        print('Ensemble accuracy of AdaptText is {0} %.'.format(ensemble_accuracy))

        ensemble_error = error_rate(ensemble_predictions, y_values)
        print('Error rate of AdaptText is {0} %.'.format(ensemble_error))

        # ROC_Curve Computation
        probabilities = np.exp(ensemble_predictions[:, 1])
        false_positive_rate, true_positive_rate, threshold_vals = roc_curve(y_values, probabilities, pos_label=1)

        # Compute ROC area
        roc_area = auc(false_positive_rate, true_positive_rate)
        print('ROC_Area of AdaptText is {0}'.format(roc_area))

        xlim = [-0.01, 1.0]
        ylim = [0.0, 1.01]

        # Add Graph Details
        plt.figure()
        plt.plot(false_positive_rate, true_positive_rate, color='darkorange', label='Ensemble Classifier : ROC curve (area = %0.2f)' % roc_area)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim(xlim)
        plt.ylim(ylim)
        # Add X and Y axes
        plt.xlabel('False Positive(FP) Rate')
        plt.ylabel('True Positive(TP) Rate')
        # Add Graph title
        plt.title('Receiver Operating Characteristic')
        # Add Graph legend
        plt.legend(loc="lower right")

        # use learn_clas_fwd for the ensemble to get close confusion matrix for the actual
        classification_interpretation = ClassificationInterpretation(learn, ensemble_predictions, y_values, losses)
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
        model_predictions, y_vals, losses = learn.get_preds(with_loss=True)

        model_accuracy = accuracy(model_predictions, y_vals)
        print('Accuracy of AdaptText is {0} %.'.format(model_accuracy))

        return model_accuracy
