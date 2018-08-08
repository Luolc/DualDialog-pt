dev_size = 1000

if __name__ == '__main__':
    with open('dialogues_text.txt', 'r') as f:
        conversations = [line.split(' __eou__')[:-1] for line in f]

    with open('valid.txt', 'w+') as f:
        for conv in conversations[:dev_size]:
            for u in conv:
                f.write(u.strip() + '\n')

    with open('test.txt', 'w+') as f:
        for conv in conversations[dev_size:dev_size * 2]:
            for u in conv:
                f.write(u.strip() + '\n')

    with open('train.txt', 'w+') as f:
        for conv in conversations[dev_size * 2:]:
            for u in conv:
                f.write(u.strip() + '\n')
