from moteur_id3.noeud_de_decision import NoeudDeDecision
from moteur_id3.id3 import ID3
from moteur_id3.noeud_de_decision_advanced import NoeudDeDecision_advanced
from moteur_id3.id3_advanced import ID3_advanced
import pandas


class ResultValues:

    #fonctions nécessaires aux tâches 3 et 4
    def diagnostic(self, user_data, healthy_rules, verbose=True, modifiable_traits=2):
        """
        :param user_data: the list of the patient's traits
        :param healthy_rules: the list of the rules ensuring a patient is healthy
        :param verbose: if true, will print a recommendation. Enabled by default
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
                print("Curable in {} change(s) by applying the following rule :".format(minimal_change))
                self.print_rule(best_rule)
            return 1
        else:
            if verbose:
                print("Not curable, it would require {} change(s) to apply the following rule :".format(minimal_change))
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

    # Fonctions nécessaires à la tâche 2
    def prediction(self, test_data):
        correct = 0

        for d in test_data:
            correct += int(self.arbre.classifie_type(d[1]) == d[0])
        correct *= 100 / len(test_data)

        print('Pourcentage de prédiction correcte: {}%'.format(correct))
        print('Pourcentage de prédiction incorrecte: {}%'.format(100 - correct))

    def import_data(self, test_data):
        test_data = pandas.read_csv(test_data).applymap(str)
        result = []

        for index, row in test_data.iterrows():
            dic = test_data.loc[index, test_data.columns != 'target'].to_dict()
            result.append([row['target'], dic])

        return result

    # Fonction nécessaire à la tâche 5
    def prediction_advance(self, test_data):
        correct = 0

        for d in test_data:
            correct += int(self.arbre_advance.classifie(d[1])[-3] == d[0][0])
        correct *= 100 / len(test_data)

        print('Pourcentage de prédiction correcte: {}%'.format(correct))
        print('Pourcentage de prédiction incorrecte: {}%'.format(100 - correct))

    def import_data_advance(self, test_data):
        test_data = pandas.read_csv(test_data).applymap(float)
        result = []

        for index, row in test_data.iterrows():
            dic = test_data.loc[index, test_data.columns != 'target'].to_dict()
            result.append([str(row['target']), dic])

        return result

    def __init__(self):

        # Do computations here

        # Task 1
        train_data = self.import_data('train_bin.csv')
        self.arbre = ID3().construit_arbre(train_data)
        # print('Decision tree :')
        # print(self.arbre)

        # Task 2
        test_data = self.import_data('test_public_bin.csv')
        if True:
            self.prediction(test_data)
        # Sanity check Task 2
        if False:
            self.prediction(train_data)

        # Task 3 and Task 4

        # compute and show each rule
        self.rules = self.arbre.calcule_regles()
        self.healty_rules = self.get_healthy_rules(self.rules)
        # print("\nDisplay all possible rules")
        # for rule in self.rules: self.print_rule(rule)

        # also shows rules on demand for a certain data
        self.faits_initiaux = train_data
        sample = self.faits_initiaux[11] # indicate here the example you want
        print("\nThe specified patient has the following data:")
        print(sample)
        print("It can be described using the following rule:")
        self.print_rule(self.arbre.calcule_regle_unique(sample[1]))
        self.diagnostic(sample[1], self.healty_rules)

        # compute how many patients we can cure with 2 changes or less
        total = [0, 0, 0]
        change_amount = 2
        for data in test_data:
            total[self.diagnostic(data[1], self.healty_rules, False, change_amount)] += 1
        print("\nWith {} changes or less, we can cure {} patients; we cannot heal {} and {} are already healthy".format(
            change_amount, total[1], total[0], total[2]))

        # Task 5
        train_data_advance = self.import_data_advance('train_continuous.csv')
        self.arbre_advance = ID3_advanced().construit_arbre(train_data_advance)
        # print('Arbre de décision :')
        # print(self.arbre_advance)

        test_data_advance = self.import_data_advance('test_public_continuous.csv')
        if True:
            self.prediction_advance(test_data_advance)
        # Sanity Check Task 5
        if False:
            self.prediction_advance(train_data_advance)

    def get_results(self):
        return [self.arbre, self.faits_initiaux, self.rules, self.arbre_advance]


r = ResultValues()
