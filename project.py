from moteur_id3.noeud_de_decision import NoeudDeDecision
from moteur_id3.id3 import ID3
import pandas


class ResultValues():
    def prediction(self,test_data):
        # Fonction nécessaire à la tâche 2
        test_data = pandas.read_csv(test_data)
        test_data = test_data.applymap(str)
        donnees_test=[]
        for index, row in test_data.iterrows():
            rd=test_data.loc[index, test_data.columns != 'target']
            dic=rd.to_dict()
            #print(dic)
            donnees_test.append([row['target'],dic])
        #print(donnees_test)
        correct=0
        unclassified=0
        for d in donnees_test:
            a=self.arbre.classifie_type(d[1])
            b=d[0]
            #print(a)
            #print(b)
            correct+=int(a==b)
            #unclassified+=int(self.arbre.classifie_type(d[1])=='_Not_enough_training_data_to_classify')
        correct/=len(donnees_test)
        #unclassified/=len(donnees_test)
        print('Pourcentage de prédiction correcte: {}%'.format(correct*100))
        #print('Pourcentage de prédiction impossible: {}%'.format(unclassified*100))
        print('Pourcentage de prédiction incorrecte: {}%'.format((1-correct-unclassified)*100))

    def __init__(self):
        
        # Do computations here
        
        # Task 1
        train_data = pandas.read_csv('train_bin.csv')
        #print(train_data)
        train_data = train_data.applymap(str)
        donnees=[]
        for index, row in train_data.iterrows():
            rd=train_data.loc[index, train_data.columns != 'target']
            dic=rd.to_dict()
            #print(dic)
            donnees.append([row['target'],dic])
        id3 = ID3()
        self.arbre = id3.construit_arbre(donnees)
        print('Arbre de décision :')
        print(self.arbre)

        # Task 2
        self.prediction('test_public_bin.csv')
        
        # Sanity Check Task 2
        if False:
            self.prediction('train_bin.csv')
        
        self.arbre = None
        # Task 3
        self.faits_initiaux = None
        self.regles = None
        # Task 5
        self.arbre_advance = None

    def get_results(self):
        return [self.arbre, self.faits_initiaux, self.regles, self.arbre_advance]
r=ResultValues()