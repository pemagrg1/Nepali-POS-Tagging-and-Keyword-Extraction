# Nepali-POS-Tagging-and-Keyword-Extraction
Nepali is the language spoken by the people of Nepal. Nepali is actually written with the Devanagari alphabet and is an Indo-Aryan Language. The Devanagari script, which is generally known as Nagari, is written from left to right. The order of the letters made up of vowels and consonants is known as the "varnamala" which means the "garland of flowers." In the Unicode Conventional, the Devanagari is constituted in three blocks. U+0900–U+097F comprises the Devanagari, U+1CD0–U+1CFF comprises the Devanagari Extended, and U+A8E0–U+A8FF comprises the Vedic Extension. 


The paper, "Structure of Nepali Grammar" by Bal Krishna Bal has an awesome explanation on the grammar of Nepali [1] where he explains how each part of speech is used in Nepali. Asmita (Student of Bal Krishna Bal) has also done her degree project under the guidance of Bal Krishna Bal on "Part of Speech Tagger for Nepali Text using SVM" where she got an accuracy of 88% [2]. Tej Bahadur Shahi,Tank Nath Dhamala, and Bikash Balami also published a paper on "Support Vector Machines based Part of Speech Tagging for Nepali Text" where they got an accuracy of 90% on TNT and 90% on SVM, using 80000 training data size[3].


Nepali and Hindi are quite similar as they both follow the Devanagari script.


Example:

English: I will go home

Hindi: मे घर जाऊंगा

Nepali: म घर जानेछु

As we can see that in Nepali and Hindi, the word "home" is same i.e. "घर" and both gives the POS tag as "NN". So, same way lets implement the Nepali POS Tagger using TNT model just like we did for Hindi POS. Lets Start!


Let's say we have a text to tag

`text = "१० वर्षीया बालिका बलात्कारपछि हत्या गर्ने सार्वजनिक"`


Let's use our own tagged data and train it.

`from nltk.tag import tnt
from nltk.corpus import indian
train_data = indian.tagged_sents('nepali.pos')
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data)`


Let's Tag the text now!


`tagged_words = (tnt_pos_tagger.tag(nltk.word_tokenize(text)))
print(tagged_words)`


[OUTPUT]:
 `[('१०', 'CD'), ('वर्षीया', 'JJ'), ('बालिका', 'NN'), ('बलात्कारपछि', 'IN'), ('हत्या', 'NN'), ('गर्ने', 'VBNE'), ('सार्वजनिक', 'JJ')]`


easy wasn't it?


The main issue here is that the its difficult to get nepali tagged corpus. So we tend to get the tag "Unk" most of the time while tagging the words ex: ('वाशिंग', 'Unk'), ('मशीन', 'Unk').


#### How to overcome that?

1. we can add more tagged sentences to nepali.pos
2. we can also use Google translator to translate and get the tag

I have tried using Google Translator API to handle the "Unk" tags by translating and getting the tags and then appending it to the nepali Corpus which gave a pretty good result. You can check in my Github.

**REF:**

[1] http://www.panl10n.net/english/outputs/Working%20Papers/Nepal/Microsoft%20Word%20-%207_E_N_396.pdf

[2] https://github.com/asmitasubedi/Nepali-Pos-Tagger/blob/master/AsmitaS_Parts%20of%20Speech%20Tagger%20for%20Nepali%20Text%20Using%20SVM.pdf

[3] https://pdfs.semanticscholar.org/b365/05276ef839d9b0cf193c6e65032ca5c73b37.pdf
