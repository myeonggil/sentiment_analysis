import os

pos_index = 1
neg_index = 1

for i in range(1, 1400000):
    if os.path.exists('.././crawling/train_movie_review/train_movie_review%d.txt' % i):
        f = open('.././crawling/train_movie_review/train_movie_review%d.txt' % i, 'r', encoding='utf-8')
        review_data = f.readline()
        dif = str(review_data).split('*'*10)

        if len(dif) != 2: continue

        if dif[0] == '' or dif[1] == '' or dif[0] == '5' or dif[1] == '5': continue

        if len(dif[0]) == 1:
            if dif[0] == '1':
                pos_review = open('.././crawling/train_pos_review/train_pos_review%d.txt' % pos_index, 'w', encoding='utf-8')
                pos_review.write(dif[1][7:])
                pos_index += 1
                pos_review.close()
            else:
                neg_review = open('.././crawling/train_neg_review/train_neg_review%d.txt' % neg_index, 'w', encoding='utf-8')
                neg_review.write(dif[1][7:])
                neg_index += 1
                neg_review.close()
        else:
            if dif[1] == '1':
                pos_review = open('.././crawling/train_pos_review/train_pos_review%d.txt' % pos_index, 'w', encoding='utf-8')
                pos_review.write(dif[1][7:])
                pos_index += 1
                pos_review.close()
            else:
                neg_review = open('.././crawling/train_neg_review/train_neg_review%d.txt' % neg_index, 'w', encoding='utf-8')
                neg_review.write(dif[1][7:])
                neg_index += 1
                neg_review.close()