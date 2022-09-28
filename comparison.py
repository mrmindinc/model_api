from sentence_transformers import util

class Comparator:
    
    def __call__(self, statement_a, statement_b):
        return self.compare(statement_a, statement_b)

    def compare(self, statement_a, statement_b):
        return 0

class CosineSimilarity(Comparator):
    
    def compare(self, text, other_text):

        similarity = util.pytorch_cos_sim(text, other_text)[0]

        return similarity
    
cosine_similarity_check = CosineSimilarity()