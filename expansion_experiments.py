from modules.expansion import FooocusExpansion

expansion = FooocusExpansion()

text = 'apple'

for i in range(64):
    print(expansion(text, seed=i))
