from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier


class SupervisedClassifier:
    class AdaBoostClassifier:

        def get_model(self,X_train,Y_train):

            ada_boost_classifier = self.get_classifier()
            ada_boost_model = ada_boost_classifier.fit(X_train,Y_train)

            exit(0)
            print("fit complete");
            return ada_boost_model


        def get_classifier(self,base_estimator='svc',estimators=None,n_estimators=50):

            base_estimator = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0,
                                       multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                                       class_weight='balanced', verbose=0, random_state=0, max_iter=1000)
            ada_boost_classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50,
                                                      algorithm='SAMME')
            return ada_boost_classifier

        def classify(self,X_train,Y_train,X_test,Y_test=None,base_estimator='svc',estimators=None,n_estimators=50):

            ada_boost_model = self.get_model(X_train,Y_train)

            train_Y_pred = ada_boost_model.predict(X_train)

            test_Y_pred = ada_boost_model.predict(X_test)

            train_acc = accuracy_score(y_pred=train_Y_pred,y_true=Y_train)

            if Y_test is not None:
                test_acc = accuracy_score(y_pred=test_Y_pred, y_true=Y_test)

                return test_Y_pred, train_acc, test_acc

            return test_Y_pred, train_acc

    class SvcClassifier:

        def get_model(self,X_train,Y_train):

            svc_model = self.get_classifier().fit(X_train,Y_train)

            return svc_model

        def get_classifier(self):

            svc_classifier = OneVsRestClassifier(LinearSVC(random_state=0))

            return svc_classifier

        def classify(self,X_train, Y_train, X_test=None, Y_test=None):

            svc_classifier = self.get_classifier()

            svc_model = svc_classifier.fit(X_train, Y_train)

            train_Y_pred = svc_model.predict(X_train)

            train_accuracy = accuracy_score(Y_train, train_Y_pred)

            test_Y_pred = svc_model.predict(X_test)

            if Y_test is not None:

                test_accuracy = accuracy_score(Y_test, test_Y_pred)

                return test_Y_pred, train_accuracy, test_accuracy

            return test_Y_pred, train_accuracy

    class LogisticRClassifier:

        def get_model(self,X_train,Y_train):

            logistic_r_model = self.get_classifier().fit(X_train,Y_train)

            return logistic_r_model

        def get_classifier(self):

            logistic_r_classifier = OneVsRestClassifier(LogisticRegression(), n_jobs=-1)

            return logistic_r_classifier


        def classify(self, X_train, Y_train, X_test=None, Y_test=None):

            logistic_r_classifier = self.get_classifier()

            logistic_r_model = logistic_r_classifier.fit(X_train, Y_train)

            train_Y_pred = logistic_r_model.predict(X_train)

            train_accuracy = accuracy_score(Y_train, train_Y_pred)

            test_Y_pred = logistic_r_model.predict(X_test)

            if Y_test is not None:

                test_accuracy = accuracy_score(Y_test, test_Y_pred)

                return test_Y_pred, train_accuracy, test_accuracy

            return test_Y_pred, train_accuracy
