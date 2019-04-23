f = open('abc.txt', 'r', encoding='utf8')
data = f.readlines()
num = 0
for i in data:
    if 'likes: ' in i:
        a = i.split('\n')
        b = a[0].split('likes: ')
        num += int(b[1])

print(num)