# wsd
NLP Project
## SOURCES
### CLARIN SVN REPO URL
> svn co http://svn.clarin-pl.eu/svn/ijn_students clarin


## GS TASK TO PARSE
> http://tools.clarin-pl.eu/share/embeddings/kgr10.plain.lemma.skipgram.dim300.neg10.vec
> clarin/autoextend.7z


# **ALGO**

Our WSD algorithm takes sentences as input and outputs a preferred sense for each polysemous word.
Given a sentence **w1 . . . wi** of i words, we retrieve a set of word senses from the sense inventory for each
**word w**. Then, for each **sense s** of **each word w**, we consider the similarity of its lexeme (the combination
of a word and one of its senses (Rothe and Schutze, 2015) with the context and the similarity of the gloss ¨
with the context.
For each potential sense s of word w, the cosine similarity is computed between its **gloss vector Gs**
and its **context vector Cw** and between the context vector Cw and the **lexeme vector Ls,w**. The score of
a given word w and sense s is thus defined as follows:

> Score(s, w) = cos(Gs, Cw) + cos(Ls,w, Cw)

The sense with the highest score is chosen. When no gloss is found for a given sense, only the second
part of the equation is used.
Prior to disambiguation itself, we sort the words by the number of senses is has, in order that the word
with the fewest senses will be considered first. The idea behind this is that words that have fewer senses
are easier to disambiguate (Chen et al., 2014). As the algorithm relies on the words in the context which
may themselves be ambiguous, if words in the context have been disambiguated already, this information
can be used for the ambiguous words that follow. We, therefore, use the resulting sense of each word for
the disambiguation of the following words starting with the “easiest” words.
Our method requires lexeme embeddings Ls,w for each sense s. For this, we use AutoExtend (Rothe
and Schutze, 2015) to create additional embeddings for senses from WordNet on the basis of word ¨
embeddings. AutoExtend is an auto-encoder that relies on the relations present in WordNet to learn
embeddings for senses and lexemes. To create these embeddings, a neural network containing lexemes
and sense layers is built, while the WordNet relations are used to create links between each layer. The
advantage of their method is that it is flexible: it can take any set of word embeddings and any lexical
database as input and produces embeddings of senses and lexemes, without requiring any extra training
data.
Ultimately, for each word w we need a vector for the context Cw, and for each sense s of word
w we need a gloss vector Gs. The context vector Cw is defined as the mean of all the content word
representations in the sentence: if a word in the context has already been disambiguated, we use the
corresponding sense embedding; otherwise, we use the word embedding. For each sense s, we take its
gloss as provided in WordNet. In line with Banerjee and Pedersen (2002), we expand this gloss with the
glosses of related meanings, excluding antonyms. Similar to the creation of the context vectors, the gloss
vector Gs is created by averaging the word embeddings of all the content words in the gloss.

## PSEUDOCODE
$$ \arg\max $$



> **W** - sentence \
> **w** - word \
> **S_w** - senses of word **w** \
> **s_w** - sense of word **w**