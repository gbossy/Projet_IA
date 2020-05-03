from moteur_id3.noeud_de_decision import NoeudDeDecision
from moteur_id3.id3 import ID3
import pandas


class ResultValues():

    def __init__(self):
        
        # Do computations here
        
        # Task 1
        raw_data = pandas.read_csv('train_bin.csv')
        #print(raw_data)
        raw_data = raw_data.applymap(str)
        donnees=[]
        for index, row in raw_data.iterrows():
            rd=raw_data.loc[index, raw_data.columns != 'target']
            dic=rd.to_dict()
            #print(dic)
            donnees.append([row['target'],dic])
        id3 = ID3()
        self.arbre = id3.construit_arbre(donnees)
        print('Arbre de décision :')
        print(self.arbre)

        self.arbre = None
        # Task 3
        self.faits_initiaux = None
        self.regles = None
        # Task 5
        self.arbre_advance = None

    def get_results(self):
        return [self.arbre, self.faits_initiaux, self.regles, self.arbre_advance]
r=ResultValues()