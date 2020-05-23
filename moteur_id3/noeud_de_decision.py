class NoeudDeDecision:
    """ Un noeud dans un arbre de décision. 
    
        This is an updated version from the one in the book (Intelligence Artificielle par la pratique).
        Specifically, if we can not classify a data point, we return the predominant class (see lines 53 - 56). 
    """

    def __init__(self, attribut, donnees, p_class, enfants=None):
        """
            :param attribut: l'attribut de partitionnement du noeud (``None`` si\
            le noeud est un noeud terminal).
            :param list donnees: la liste des données qui tombent dans la\
            sous-classification du noeud.
            :param enfants: un dictionnaire associant un fils (sous-noeud) à\
            chaque valeur de l'attribut du noeud (``None`` si le\
            noeud est terminal).
        """

        self.attribut = attribut
        self.donnees = donnees
        self.enfants = enfants
        self.p_class = p_class

    def terminal(self):
        """ Vérifie si le noeud courant est terminal. """

        return self.enfants is None

    def classe(self):
        """ Si le noeud est terminal, retourne la classe des données qui\
            tombent dans la sous-classification (dans ce cas, toutes les\
            données font partie de la même classe. 
        """

        if self.terminal():
            return self.donnees[0][0]

    def calcule_regles(self, prefixe=""):
        """
        <>
        :return:
        """
        if self.terminal():
            return [prefixe + 'Alors {}'.format(self.classe().upper())]
        else:
            resultat = []
            for valeur, enfant in self.enfants.items():
                for sucessor in enfant.calcule_regles(prefixe + 'Si {} = {}, '.format(self.attribut, valeur)):
                    resultat.append(sucessor)
            return resultat

    def calcule_regle_unique(self, donnee):
        """ Calcule la regle logique permettant de classifier la donnee speficiee
            :param donnee: la donnee a classifier.
            :return: la regle correspondante
        """
        rep = ''
        if self.terminal():
            rep += 'Alors {}'.format(self.classe().upper())
        else:
            valeur = donnee[self.attribut]
            rep += 'Si {} = {}, '.format(self.attribut, valeur.upper())
            try:
                rep += self.enfants[valeur].calcule_regle_unique(donnee)
            except:
                rep += self.p_class
        return rep

    def classifie_type(self, donnee):
        """ Classifie une donnée à l'aide de l'arbre de décision duquel le noeud\
            courant est la racine.

            :param donnee: la donnée à classifier.
            :return: la classe de la donnée selon le noeud de décision courant.
        """
        if self.terminal():
            return self.classe()
        else:
            valeur = donnee[self.attribut]
            try:
                return self.enfants[valeur].classifie_type(donnee)
            except:
                return self.p_class

    def repr_arbre(self, level=0):
        """ Représentation sous forme de string de l'arbre de décision duquel\
            le noeud courant est la racine. 
        """

        rep = ''
        if self.terminal():
            rep += '---'*level
            rep += 'Alors {}\n'.format(self.classe().upper())
            rep += '---'*level
            rep += 'Décision basée sur les données:\n'
            for donnee in self.donnees:
                rep += '---'*level
                rep += str(donnee) + '\n' 

        else:
            for valeur, enfant in self.enfants.items():
                rep += '---'*level
                rep += 'Si {} = {}: \n'.format(self.attribut, valeur.upper())
                rep += enfant.repr_arbre(level+1)

        return rep

    def __repr__(self):
        """ Représentation sous forme de string de l'arbre de décision duquel\
            le noeud courant est la racine. 
        """

        return str(self.repr_arbre(level=0))