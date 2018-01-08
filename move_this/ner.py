import nltk
tokens = nltk.word_tokenize('''"""

 Your opinion here. 
 
 My name is Wong.
 
 www@123

Your opinion would be most welcome at Wikipedia talk:WikiProject United States courts and judges#Proposed renaming of List of judicial appointments made by Barack Obama. Cheers!  T """''')
tokens = nltk.pos_tag(tokens)
print(tokens)

tree = nltk.ne_chunk(tokens, binary=True)
for tree in nltk.ne_chunk(tokens, binary=True).subtrees():
        # 过滤根树
        if tree.label() == "S":
            continue
        else:
            for t in tree:
                print(t)


