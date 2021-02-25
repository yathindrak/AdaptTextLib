class Predictor():
    def __init__(self, learn_clas_fwd, learn_clas_bwd, classes):
        self.learn_clas_fwd = learn_clas_fwd
        self.learn_clas_bwd = learn_clas_bwd
        self.classes = classes

    def predict(self, inputText):
        f_pred = self.learn_clas_fwd.predict(inputText)[2]
        b_pred = self.learn_clas_bwd.predict(inputText)[2]
        average_prediction = f_pred + b_pred

        max_value = average_prediction[0]
        max_index = 0

        for i, x in enumerate(average_prediction):
            if x > max_value:
                max_value = x
                max_index = i

        return self.classes[max_index]