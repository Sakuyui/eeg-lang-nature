
class LanguageConfiguration(object):
    def __init__(self, configuration):
        self.configuration = configuration

    def configuration_word_count(self):
        return self.configuration['word_list_size']

    