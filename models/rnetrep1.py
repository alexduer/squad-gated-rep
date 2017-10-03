from models.rnetrep0 import RnetRep0


class RnetRep1(RnetRep0):
    def encoding_layers(self):
        par_encoded = self.apply_dropout(self.encoding_layer(self.par_vectors, self.par_num_words, False))
        qu_encoded = self.apply_dropout(self.encoding_layer(self.qu_vectors, self.qu_num_words, True))
        return par_encoded, qu_encoded
