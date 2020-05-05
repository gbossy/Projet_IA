from math import log
from .noeud_de_decision import NoeudDeDecision

class ID3:
    """ Algorithme ID3. """
    
    def construit_arbre(self, donnees):
        """ Construit un arbre de décision à partir des données d'apprentissage.

            :param list donnees: les données d'apprentissage\
            ``[classe, {attribut -> valeur}, ...]``.
            :return: une instance de NoeudDeDecision correspondant à la racine de\
            l'arbre de décision.
        """
        
        # Nous devons extraire les domaines de valeur des 
        # attributs, puisqu'ils sont nécessaires pour 
        # construire l'arbre.
        attributs = {}
        for donnee in donnees:
            for attribut, valeur in donnee[1].items():
                valeurs = attributs.get(attribut)
                if valeurs is None:
                    valeurs = set()
                    attributs[attribut] = valeurs
                valeurs.add(valeur)
            
        arbre = self.construit_arbre_recur(donnees, attributs)
        
        return arbre

    def construit_arbre_recur(self, donnees, attributs):
        """ Construit rédurcivement un arbre de décision à partir 
            des données d'apprentissage et d'un dictionnaire liant
            les attributs à la liste de leurs valeurs possibles.

            :param list donnees: les données d'apprentissage\
            ``[classe, {attribut -> valeur}, ...]``.
            :param attributs: un dictionnaire qui associe chaque\
            attribut A à son domaine de valeurs a_j.
            :return: une instance de NoeudDeDecision correspondant à la racine de\
            l'arbre de décision.
        """
        if len(donnees)==0:
            return None
        else:
            bool_=True
            c=donnees[0][0]
            for d in donnees:
                if d[0]!=c:
                    bool_=False
            if bool_:
                return NoeudDeDecision(None,donnees)
            else:
                H={}
                a_min=list(attributs.keys())[0]
                min_=self.h_C_A(donnees, a_min,attributs.get(a_min))
                for a in attributs.keys():
                    H[a]=self.h_C_A(donnees, a,attributs.get(a))
                    if H[a]<min_:
                        a_min=a
                        min_=H[a]
                valeurs=attributs.get(a_min)
                part=self.partitionne(donnees,a_min,valeurs)
                enf={}
                new_att=attributs.copy()
                new_att.pop(a_min)
                for v in valeurs:
                    enf[v]=self.construit_arbre_recur(part.get(v),new_att)
                return NoeudDeDecision(a_min,donnees,enf)
                


    def partitionne(self, donnees, attribut, valeurs):
        """ Partitionne les données sur les valeurs a_j de l'attribut A.

            :param list donnees: les données à partitioner.
            :param attribut: l'attribut A de partitionnement.
            :param list valeurs: les valeurs a_j de l'attribut A.
            :return: un dictionnaire qui associe à chaque valeur a_j de\
            l'attribut A une liste l_j contenant les données pour lesquelles A\
            vaut a_j.
        """
        dic={}
        #print(attribut)
        #print(valeurs)
        for a in valeurs:
            dic[a]=[]
        for d in donnees:
            #print(d)
            #print(dic[d[1].get(attribut)])
            dic[d[1].get(attribut)].append(d)
        return dic

    def p_aj(self, donnees, attribut, valeur):
        """ p(a_j) - la probabilité que la valeur de l'attribut A soit a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.            
            :return: p(a_j)
        """
        p=0
        for d in donnees:
            p+=int(d[1].get(attribut)==valeur)
        return p/len(donnees)

    def p_ci_aj(self, donnees, attribut, valeur, classe):
        """ p(c_i|a_j) - la probabilité conditionnelle que la classe C soit c_i\
            étant donné que l'attribut A vaut a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :param classe: la valeur c_i de la classe C.
            :return: p(c_i | a_j)
        """
        p=0
        c=0
        for d in donnees:
            if d[1].get(attribut)==valeur:
                c+=1
                p+=int(d[0]==classe)
        if c!=0:
            return p/c
        else:
            #print('c=0')
            return 0

    def h_C_aj(self, donnees, attribut, valeur):
        """ H(C|a_j) - l'entropie de la classe parmi les données pour lesquelles\
            l'attribut A vaut a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :return: H(C|a_j)
        """
        s=0
        classes=[]
        for d in donnees:
            classes.append(d[0])
        for c in classes:
            p=self.p_ci_aj(donnees, attribut, valeur, c)
            if p!=0:
                s+=p*log(p,2)
        return - s

    def h_C_A(self, donnees, attribut, valeurs):
        """ H(C|A) - l'entropie de la classe après avoir choisi de partitionner\
            les données suivant les valeurs de l'attribut A.
            
            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param list valeurs: les valeurs a_j de l'attribut A.
            :return: H(C|A)
        """
        s=0
        for v in valeurs:
            s+=self.h_C_aj(donnees, attribut, v)
        return s