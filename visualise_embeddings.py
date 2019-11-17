from src.embeder import Embeder
from src.models import *
import itertools
import os


if __name__ == '__main__':
    # Load corpus
    corpus_dir = './categorization/learningData/'
    x = Embeder(corpus_dir)
    
    # Load model
    model_dir = './models/elmo_polish/'
    model = ElmoModel(model_dir)

    # Compute embeddings
    contexts = ['one-mention', 'document', 'corpus']
    neighborhoods = [2, 3, 4, 5, 'sentence']
    for context, neighborhood in list(itertools.product(contexts, neighborhoods))[10:]:  # modify range
        save_dir = f'../projections/elmo_{context}_{str(neighborhood)}/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        embeds = x.get_embeddings(context, neighborhood, model)
        x.save_to_tensorboard(embeds, save_dir)
