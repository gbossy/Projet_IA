from moteur_id3.noeud_de_decision import NoeudDeDecision
from moteur_id3.id3 import ID3
import pandas


class ResultValues:

    def affiche_regles(self):
        regles = self.arbre.calcule_regles()

        for regle in regles:
            *conditions, resultat = regle
            affichage = ''
            for condition in conditions :
                affichage += 'Si {} = {}, '.format(condition[0], condition[1])
            affichage += 'alors {}'.format(resultat[1])
            print(affichage)

    def importe_donnees(self, test_data):
        test_data = pandas.read_csv(test_data).applymap(str)
        donnees_test = []

        for index, row in test_data.iterrows():
            dic = test_data.loc[index, test_data.columns != 'target'].to_dict()
            donnees_test.append([row['target'], dic])

        return donnees_test

    # Fonction nécessaire à la tâche 2
    def prediction(self, donnees_test):
        correct = 0
        for d in donnees_test:
            correct += int(self.arbre.classifie_type(d[1]) == d[0])

        return correct*100 / len(donnees_test)

    def __init__(self):

        # Do computations here

        # Task 1
        donnees_entrainement = self.importe_donnees('train_bin.csv')
        self.arbre = ID3().construit_arbre(donnees_entrainement)

        print('Arbre de décision :')
        print(self.arbre)

        # Task 2
        donnees_test = self.importe_donnees('test_public_bin.csv')
        self.resultat = self.prediction(donnees_test)

        print('Pourcentage de prédictions correctes / incorrectes  : {}%'.format(self.resultat) +
              ' / {}%'.format(100-self.resultat))

        # Task 3

        self.affiche_regles()
        # la ligne suivante affiche la regle correspondante à chaque échantillon de donnée
        # for donnee in donnees_test: print(self.arbre.calcule_regle_unique(donnee[1]))

        # Task 4
        self.faits_initiaux = None
        self.regles = None
        # Task 5
        self.arbre_advance = None

    def get_results(self):
        return [self.arbre, self.faits_initiaux, self.regles, self.arbre_advance]


r = ResultValues()
