from math import log
from .noeud_de_decision_advanced import NoeudDeDecision_advanced

class ID3_advanced:
    """ Algorithme ID3. 

        This is an updated version from the one in the book (Intelligence Artificielle par la pratique).
        Specifically, in construit_arbre_recur(), if donnees == [] (line 70), it returns a terminal node with the predominant class of the dataset -- as computed in construit_arbre() -- instead of returning None.
        Moreover, the predominant class is also passed as a parameter to NoeudDeDecision().
    """
    
    def construit_arbre(self, donnees):
        """ Construit un arbre de décision à partir des données d'apprentissage.

            :param list donnees: les données d'apprentissage\
            ``[classe, {attribut -> valeur}, ...]``.
            :return: une instance de NoeudDeDecision correspondant à la racine de\
            l'arbre de décision.
        """

        # Find the predominant class
        classes = set([row[0] for row in donnees])
        predominant_class_counter = -1
        for c in classes:
            if [row[0] for row in donnees].count(c) >= predominant_class_counter:
                predominant_class_counter = [row[0] for row in donnees].count(c)
                predominant_class = c            
        arbre = self.construit_arbre_recur(donnees, predominant_class)

        return arbre

    def construit_arbre_recur(self, donnees, predominant_class):
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
        
        def classe_unique(donnees):
            """ Vérifie que toutes les données appartiennent à la même classe. """
            
            if len(donnees) == 0:
                return True 
            premiere_classe = donnees[0][0]
            for donnee in donnees:
                if donnee[0] != premiere_classe:
                    return False 
            return True

        if donnees == []:
            return NoeudDeDecision_advanced(None, [str(predominant_class), dict()], str(predominant_class))

        # Si toutes les données restantes font partie de la même classe,
        # on peut retourner un noeud terminal.         
        elif classe_unique(donnees):
            return NoeudDeDecision_advanced(None, donnees, str(predominant_class))
            
        else:
            attributs_restants = {}
            for donnee in donnees:
                for attribut, valeur in donnee[1].items():
                    valeurs = attributs_restants.get(attribut)
                    if valeurs is None:
                        valeurs = set()
                        attributs_restants[attribut] = valeurs
                    valeurs.add(valeur)
            attributs = attributs_restants.copy()
            l=[]
            for a in attributs.keys():
                if len(attributs[a])==1:
                    l.append(a)
                elif len(attributs[a])==0:
                    print('Erreur: attribut sans valeurs')
            for e in l:
                del attributs[e]
            # Sélectionne l'attribut qui réduit au maximum l'entropie.
            ####On doit choisir parmi tous les splits plutôt que juste un argument
            h_C_As_attribs = [(self.h_C_A(donnees, attribut, attributs[attribut]), 
                               attribut) for attribut in attributs]

            val,attribut = min(h_C_As_attribs, key=lambda h_a: h_a[0][1])

            # Crée les sous-arbres de manière récursive.
            #####On doit retirer un argument uniquement s'il ne reste plus qu'une seule/2 valeur pour cet argument
            partitions = self.partitionne(donnees, attribut, attributs[attribut])
            
            enfants = {}
            ####On ne doit split qu'en 2 sous-arbres
            p1=None
            p2=None
            for valeur, partition in partitions.items():
                if valeur<val[0]:
                    if p1 is None:
                        p1=partition
                    else:
                        p1.extend(partition)
                else:
                    if p2 is None:
                        p2=partition
                    else:
                        p2.extend(partition)
            #On attribue la valeur du split si on est au dessus ou egal, et le min des valeurs si en dessous
            enfants['>=']=self.construit_arbre_recur(p2,predominant_class)
            enfants['<']=self.construit_arbre_recur(p1,predominant_class)

            return NoeudDeDecision_advanced(attribut, donnees, str(predominant_class),val[0], enfants)

    def partitionne(self, donnees, attribut, valeurs):
        """ Partitionne les données sur les valeurs a_j de l'attribut A.

            :param list donnees: les données à partitioner.
            :param attribut: l'attribut A de partitionnement.
            :param list valeurs: les valeurs a_j de l'attribut A.
            :return: un dictionnaire qui associe à chaque valeur a_j de\
            l'attribut A une liste l_j contenant les données pour lesquelles A\
            vaut a_j.
        """
        partitions = {valeur: [] for valeur in valeurs}
        
        for donnee in donnees:
            partition = partitions[donnee[1][attribut]]
            partition.append(donnee)
            
        return partitions

    def p_aj(self, donnees, attribut, valeur):
        """ p(a_j) - la probabilité que la valeur de l'attribut A soit a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.            
            :return: p(a_j)
        """
        # Nombre de données.
        nombre_donnees = len(donnees)
        
        # Permet d'éviter les divisions par 0.
        if nombre_donnees == 0:
            return 0.0
        
        # Nombre d'occurrences < a_j parmi les données.
        nombre_aj = 0
        for donnee in donnees:
            if donnee[1][attribut] < valeur:
                nombre_aj += 1

        # p(a_j) = nombre d'occurrences > a_j parmi les données / 
        #          nombre de données.
        return nombre_aj / nombre_donnees

    def p_ci_aj(self, donnees, attribut, valeur, classe):
        """ p(c_i|a_j) - la probabilité conditionnelle que la classe C soit c_i\
            étant donné que l'attribut A < a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :param classe: la valeur c_i de la classe C.
            :return: p(c_i | a_j)
        """
        # Nombre d'occurrences de la valeur a_j parmi les données.
        donnees_aj = [donnee for donnee in donnees if donnee[1][attribut] < valeur]
        nombre_aj = len(donnees_aj)
        
        # Permet d'éviter les divisions par 0.
        if nombre_aj == 0:
            return 0
        
        # Nombre d'occurrences de la classe c_i parmi les données pour lesquelles 
        # A vaut a_j.
        donnees_ci = [donnee for donnee in donnees_aj if donnee[0] == classe]
        nombre_ci = len(donnees_ci)

        # p(c_i|a_j) = nombre d'occurrences de la classe c_i parmi les données 
        #              pour lesquelles A vaut a_j /
        #              nombre d'occurrences de la valeur a_j parmi les données.
        return nombre_ci / nombre_aj
    def p_ci_aj2(self, donnees, attribut, valeur, classe):
        """ p(c_i|a_j) - la probabilité conditionnelle que la classe C soit c_i\
            étant donné que l'attribut A >= a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :param classe: la valeur c_i de la classe C.
            :return: p(c_i | a_j)
        """
        # Nombre d'occurrences de la valeur a_j parmi les données.
        donnees_aj = [donnee for donnee in donnees if donnee[1][attribut] >= valeur]
        nombre_aj = len(donnees_aj)
        
        # Permet d'éviter les divisions par 0.
        if nombre_aj == 0:
            return 0
        
        # Nombre d'occurrences de la classe c_i parmi les données pour lesquelles 
        # A vaut a_j.
        donnees_ci = [donnee for donnee in donnees_aj if donnee[0] == classe]
        nombre_ci = len(donnees_ci)

        # p(c_i|a_j) = nombre d'occurrences de la classe c_i parmi les données 
        #              pour lesquelles A vaut a_j /
        #              nombre d'occurrences de la valeur a_j parmi les données.
        return nombre_ci / nombre_aj

    def h_C_aj(self, donnees, attribut, valeur):
        """ H(C|a_j) - l'entropie de la classe parmi les données pour lesquelles\
            l'attribut A < a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :return: H(C|a_j)
        """
        # Les classes attestées dans les exemples.
        classes = list(set([donnee[0] for donnee in donnees]))
        
        # Calcule p(c_i|<a_j) pour chaque classe c_i.
        p_ci_ajs = [self.p_ci_aj(donnees, attribut, valeur, classe) 
                    for classe in classes]

        # Si p vaut 0 -> plog(p) vaut 0.
        return -sum([p_ci_aj * log(p_ci_aj, 2.0) 
                    for p_ci_aj in p_ci_ajs 
                    if p_ci_aj != 0])

    def h_C_aj2(self, donnees, attribut, valeur):
        """ H(C|a_j) - l'entropie de la classe parmi les données pour lesquelles\
            l'attribut A >= a_j.

            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param valeur: la valeur a_j de l'attribut A.
            :return: H(C|a_j)
        """
        # Les classes attestées dans les exemples.
        classes = list(set([donnee[0] for donnee in donnees]))
        
        # Calcule p(c_i|>=a_j) pour chaque classe c_i.
        p_ci_ajs = [self.p_ci_aj2(donnees, attribut, valeur, classe) 
                    for classe in classes]

        # Si p vaut 0 -> plog(p) vaut 0.
        return -sum([p_ci_aj * log(p_ci_aj, 2.0) 
                    for p_ci_aj in p_ci_ajs 
                    if p_ci_aj != 0])

    def h_C_A(self, donnees, attribut, valeurs):
        """ H(C|A) - l'entropie de la classe après avoir choisi de partitionner\
            les données suivant les valeurs de l'attribut A.
            
            :param list donnees: les données d'apprentissage.
            :param attribut: l'attribut A.
            :param list valeurs: les valeurs a_j de l'attribut A.
            :return: La valeur m de l'argument qui minimise l'entropie et H(C|A)
        """
        ####On veut itérer sur les séparations potentielles
        # Calcule P(a_j) pour chaque valeur a_j de l'attribut A.
        H={}
        for v in valeurs:
            H[v]=self.h_C_aj(donnees,attribut,v)*self.p_aj(donnees, attribut, v)+self.h_C_aj2(donnees,attribut,v)*(1-self.p_aj(donnees, attribut, v))
        m=min(valeurs,key=lambda x:H[x])
        return [m,H[m]]