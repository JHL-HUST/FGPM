class Config(object):
    """
    Store hyper-parameters of neural networks
    """
   
    num_classes = {'yahoo_answers': 10, 'ag_news': 4, 'dbpedia':14}
    word_max_len = {'yahoo_answers': 500, 'ag_news': 250, 'dbpedia':250}
    num_words = {'yahoo_answers': 50000, 'ag_news': 50000, 'dbpedia':50000}
    embedding_size = {'yahoo_answers': 300, 'ag_news': 300, 'dbpedia':300}

    stop_words = ['the', 'a', 'an', 'to', 'of', 'and', 'with', 'as', 'at', 'by', 'is', 'was', 'are', 'were', 'be', 'he', 'she', 'they', 'their', 'this', 'that']


    
config = Config()
