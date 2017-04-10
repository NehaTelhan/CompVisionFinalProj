import os

img_count = 0
for filename in os.listdir('../edges'):
    img_count += 1
    if os.path.isfile('../dataset/' + filename):
        print(filename)

print(img_count)