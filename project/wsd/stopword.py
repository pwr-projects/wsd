class StopWord:
    
    def __init__(self, file):
        self._stop_words = self._load_file(file)
    
    @property
    def stop_words(self):
        return self._stop_words
    
    def _load_file(self, file):
        with open(file, encoding='utf-8') as f:
            return f.read().splitlines()
    
    def is_stop_word(self, word):
        return word.lower() in self._stop_words