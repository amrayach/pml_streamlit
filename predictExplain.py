import configparser
from model import CharacterLevelCNN
import torch
from innvestigator import InnvestigateModel



args = configparser.ConfigParser()
args.read('argsConfig.ini')


class ModelsDeploy(object):
    def __init__(self):
        self.ag_news_model = CharacterLevelCNN(4, args)
        self.ag_news_model_checkpoint = torch.load('AgNewsModel1.pt', map_location=torch.device('cpu'))
        self.ag_news_model.load_state_dict(self.ag_news_model_checkpoint['state_dict'])
        self.ag_news_lrp = InnvestigateModel(self.ag_news_model, lrp_exponent=1, method="e-rule", beta=.5)

        self.yelp_model = CharacterLevelCNN(2, args)
        self.yelp_model_checkpoint = torch.load('YelpModel.pt', map_location=torch.device('cpu'))
        self.yelp_model.load_state_dict(self.yelp_model_checkpoint['state_dict'])
        self.yelp_lrp = InnvestigateModel(self.yelp_model, lrp_exponent=1, method="e-rule", beta=.5)
        self.alphabet = args.get('DataSet', 'alphabet')
        self.l0 = args.getint('DataSet', 'l0')


    def oneHotEncode(self, sentence):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = sentence[:self.l0]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char) != -1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)


    def predict_probs(self, sentence, model='yelp'):
        input_tensor = self.oneHotEncode(sentence)
        input_tensor = torch.unsqueeze(input_tensor, 0)

        if model == 'yelp':
            with torch.no_grad():
                predictions = self.yelp_model(input_tensor)
        else:
            with torch.no_grad():
                predictions = self.ag_news_model(input_tensor)


        pred = torch.max(predictions, 1)[1].cpu().numpy().tolist()[0]
        probs = torch.exp(predictions) * 100
        probs = probs.cpu().numpy().tolist()[0]

        return pred, probs





def main():
    obj = ModelsDeploy()
    a, b = obj.predict_probs("Like any Barnes & Noble, it has a nice comfy cafe, and a large selection of books.  The staff is very friendly and helpful.  They stock a decent selection, and the prices are pretty reasonable.  Obviously it's hard for them to compete with Amazon.  However since all the small shop bookstores are gone, it's nice to walk into one every once in a while.")

    print()


if __name__ == '__main__':
    main()