from moteur_id3.noeud_de_decision import NoeudDeDecision
from moteur_id3.id3 import ID3
import pandas


class ResultValues:

    def diagnostic(self, user_data, healthy_rules, verbose=1, modifiable_traits=2):
        """
        :param user_data: the list of the patient's traits
        :param healthy_rules: the list of the rules ensuring a patient is healthy
        :param modifiable_traits: the amount of traits that a patient can change (except for age and sex)
        :return: 2 if healthy, 1 if curable, 0 in not curable
        """
        best_rule = []
        minimal_change = float("inf")
        for rule in healthy_rules:
            change = 0
            *conditions, resultat = rule
            for pair in conditions:
                if user_data.get(pair[0]) != pair[1]:
                    change += float("inf") if (pair[0] == "age") or (pair[0] == "sex") else 1
            if change == 0:
                if verbose:
                    print("The patient is healthy, as shows the following rule :")
                    self.print_rule(rule)
                return 2
            elif change < minimal_change:
                minimal_change = change
                best_rule = rule

        if minimal_change <= modifiable_traits:
            if verbose:
                print("The patient can be cured in {} change(s) by applying the following rule :".format(minimal_change))
                self.print_rule(best_rule)
            return 1
        else:
            if verbose:
                print("The patient cannot be cured, it would require {} change(s)to apply the following rule :".format(minimal_change))
                self.print_rule(best_rule)
            return 0

    def print_rule(self, rule):
        *conditions, result = rule
        display = ''
        for condition in conditions:
            display += 'If {} = {}, '.format(condition[0], condition[1])
        display += 'then {}'.format(result[1])
        print(display)

    def get_healthy_rules(self, rules):
        result = []
        for rule in rules:
            if rule[-1][1] == '0':
                result.append(rule)
        return result

    def affiche_regles(self, regles):
        for regle in regles:
            self.print_rule(regle)

    def import_data(self, test_data):
        test_data = pandas.read_csv(test_data).applymap(str)
        result = []

        for index, row in test_data.iterrows():
            dic = test_data.loc[index, test_data.columns != 'target'].to_dict()
            result.append([row['target'], dic])

        return result

    # Fonction nécessaire à la tâche 2
    def prediction(self, data_test):
        correct = 0
        for d in data_test:
            correct += int(self.arbre.classifie_type(d[1]) == d[0])

        return correct * 100 / len(data_test)

    def __init__(self):

        # Do computations here

        # Task 1
        training_data = self.import_data('train_bin.csv')
        self.arbre = ID3().construit_arbre(training_data)
        print('Decision tree :')
        print(self.arbre)

        # Task 2
        donnees_test = self.import_data('test_public_bin.csv')
        self.resultat = self.prediction(donnees_test)

        print('Pourcentage de prédictions correctes / incorrectes  : {}%'.format(self.resultat) +
              ' / {}%'.format(100 - self.resultat))

        # Task 3 and Task 4

        # compute and show each rule
        self.regles = self.arbre.calcule_regles()
        self.healty_rules = self.get_healthy_rules(self.regles)
        print("\nDisplay all possible rules")
        self.affiche_regles(self.regles)

        # also shows rules on demand for a certain data
        self.faits_initiaux = donnees_test[1] # indicate here the example you want
        print("\nThe specified patient has the following data:")
        print(self.faits_initiaux)
        print("It can be described using the following rule:")
        self.affiche_regles([self.arbre.calcule_regle_unique(self.faits_initiaux[1])])
        self.diagnostic(self.faits_initiaux[1], self.healty_rules)

        # compute how many patients we can cure with 2 changes or less
        total = [0, 0, 0]
        for data in donnees_test:
            total[self.diagnostic(data[1], self.healty_rules, 0)]+=1
        print("\nWith 2 changes or less, we can cure {} patients ; we cannot heal {} and {} are already healthy".format(total[1],total[0],total[2]))

        # Task 5
        self.arbre_advance = None

    def get_results(self):
        return [self.arbre, self.faits_initiaux, self.regles, self.arbre_advance]


r = ResultValues()
