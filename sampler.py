import Levenshtein





def edit_distance(word1,word2):
    distance=Levenshtein.distance(word1,word2)
    return distance

