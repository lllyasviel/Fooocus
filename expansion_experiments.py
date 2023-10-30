from modules.expansion import FooocusExpansion

expansion = FooocusExpansion()

text = 'stone'

for i in range(64):
    print(expansion(text, seed=i))
