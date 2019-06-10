from nltk.tag import tnt
from nltk.corpus import indian
import nltk
from nltk.tree import Tree

def nepali_model():
    train_data = indian.tagged_sents('<path/to/nepali.pos>')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    return tnt_pos_tagger


def get_keywords(pos):
    grammar = r"""NP:{<NN.*>}"""
    chunkParser = nltk.RegexpParser(grammar)
    chunked = chunkParser.parse(pos)
    continuous_chunk = set()
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.add(named_entity)
                current_chunk = []
            else:
                continue
    return (continuous_chunk)

text = "१० वर्षीया बालिका बलात्कारपछि हत्या गर्ने सार्वजनिक"

model = nepali_model()
new_tagged = (model.tag(nltk.word_tokenize(text)))
print(new_tagged)
print()
print("====KEYWORDS===")
print(get_keywords(new_tagged))

